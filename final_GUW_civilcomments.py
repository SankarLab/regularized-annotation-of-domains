# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from data import group_sln, class_balance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
    for i, grp in df.groupby('target'):
        for j, subgrp in grp.groupby('group'):
            indices[subgrp.index.to_list()] = len(df)/(4 * len(df[(df['target'] == i) & (df['group'] == j)]))
    return indices
    
    
def run_exp(val_data, test_data, C):
    weights = sample_weights(val_data)
    DFR = LogisticRegression(penalty='l1', solver='liblinear', C=C, fit_intercept=True).fit(val_data.drop(['target','group'], axis=1), val_data['target'], sample_weight = weights)
    return wga(DFR,val_data), wga(DFR,test_data)

np.random.seed(0)
seeds = np.random.randint(200, size=(10)) 


datasets = ['civilcomments']
noise = [0, 0.05, 0.1, 0.15, 0.2]
c_values = [0.001129, 0.001129, 0.001129, 0.001129, 0.001129]
final_results = pd.DataFrame(columns=['dataset', 'noise', 'wga_mean', 'wga_std', 'C'])

for dataset in datasets:
    base_path = '../'+dataset+'/'


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
    
    for i, noise_level in enumerate(noise):
    
        c = c_values[i]
        dfr_wga = np.zeros(10)
           
        for i, seed in enumerate(seeds):
    
            np.random.seed(seed)

            test_data = original_val_data.sample(frac=0.5,replace=False)
            train_data = group_sln(original_val_data.drop(test_data.index),p=noise_level)
            full_train_data = pd.concat([group_sln(test_data,p=noise_level), train_data],ignore_index=True)
    
            _,dfr_wga[i] = run_exp(full_train_data, final_test_data, c)
               
               
        print("Group Upweighted LLR (" + dataset + ")(" + str(noise_level) + "): ", dfr_wga.mean(), dfr_wga.std())
        final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'wga_mean': dfr_wga.mean(), 'wga_std': dfr_wga.std(), 'C': c } 
        
    path = 'results/final_group_upweighted_' + dataset + '.csv'
    final_results.to_csv(path, mode='a', header=True)