# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from data import group_sln
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
import argparse

def wga(clf, data):
    data = data.copy()
    clf_min = np.inf
    for _, group in data.groupby(['target','group']):
        data = group.drop(['target','group'],axis=1).values
        target = group['target'].values
        clf_min = min(clf.score(data,target),clf_min)
    return clf_min

def sample_weights(val_df):
    df = val_df.copy()
    df.reset_index(inplace = True)
    indices = np.zeros(len(df))
    num_groups = len(np.unique(val_df['target'].values)) * len(np.unique(val_df['group'].values))
    for i, grp in df.groupby('target'):
        for j, subgrp in grp.groupby('group'):
            indices[subgrp.index.to_list()] = len(df)/(num_groups * len(df[(df['target'] == i) & (df['group'] == j)]))
    return indices
    
    
def run_exp(val_data, test_data, C):
    weights = sample_weights(val_data)
    DFR = LogisticRegression(penalty='l1', solver='liblinear', C=C, fit_intercept=True).fit(val_data.drop(['target','group'], axis=1), val_data['target'], sample_weight = weights)
    return wga(DFR,val_data), wga(DFR,test_data)

    
datasets = ['celebA', 'cmnist', 'wb', 'multinli', 'civilcomments']
noise = [0, 0.05, 0.1, 0.15, 0.2]
final_results = pd.DataFrame(columns=['dataset', 'noise', 'wga_mean', 'wga_std', 'C', 'exp'])
C_VALUES = np.logspace(-4,0,num=20, base=10)

for dataset in datasets:
    base_path = '/home/rayyaga2/Thesis work/OOD_analysis/mixup_exps/datasets/'+dataset+'/'
    X = np.load(base_path+dataset+'_val_embeddings.npy')
    y = np.load(base_path+dataset+'_val_labels.npy')
    group = np.load(base_path+dataset+'_val_groups.npy')
    test_X = np.load(base_path+dataset+'_test_embeddings.npy')
    test_y = np.load(base_path+dataset+'_test_labels.npy')
    test_group = np.load(base_path+dataset+'_test_groups.npy')
    
    # %%
    original_val_data = pd.DataFrame(X)
    original_val_data['target'] = y
    original_val_data['group'] = group
    
    final_test_data = pd.DataFrame(test_X)
    final_test_data['target'] = test_y
    final_test_data['group'] = test_group
    
    for noise_level in noise:
       
       for exp in range(10):
           
            print(dataset, noise_level, exp)
            
            np.random.seed(exp)
    
            
            test_data = original_val_data.sample(frac=0.5,replace=False)
            train_data = group_sln(original_val_data.drop(test_data.index),p=noise_level)
            full_train_data = pd.concat([group_sln(test_data,p=noise_level), train_data],ignore_index=True)
           
            
            results = pd.DataFrame(columns=['C', 'val_wga','test_wga','type'])
            
            DFR_best = -np.inf
            for c in C_VALUES:
                dfr_val, dfr_test = run_exp(train_data,test_data, c)
                results.loc[len(results)] = {'C':c,'alpha':None,'n':None,'val_wga':dfr_val,'test_wga':dfr_test,'type':'GUW'}
                
                
            dfr_avg_param = results[results['type'] == 'GUW'].groupby(['C'])['test_wga'].mean().idxmax()
            
            dfr_wga = np.zeros(10)

            
            for i in range(10):
                _,dfr_wga[i] = run_exp(full_train_data, final_test_data, dfr_avg_param)
               
               
            print("LLR - group upweighted (" + dataset + ")(" + str(noise_level) + "): ", dfr_wga.mean(), dfr_wga.std())
            final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'wga_mean': dfr_wga.mean(), 'wga_std': dfr_wga.std(), 'C': dfr_avg_param, 'exp': exp } 
            results = results[0:0] 

    path = 'results/hparams_GUW_' + dataset + '.csv'
    final_results.to_csv(path, mode='a', header=True)