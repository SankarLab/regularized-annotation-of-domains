# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from data import group_balance, group_sln, wga

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
    GUW = LogisticRegression(penalty='l1', solver='liblinear', C=C, fit_intercept=True).fit(val_data.drop(['target','group'], axis=1), val_data['target'], sample_weight = weights)
    return GUW.coef_, GUW.intercept_, wga(GUW,val_data), wga(GUW,test_data)
    
    
datasets = ['celebA', 'cmnist', 'wb', 'multinli', 'civilcomments']
noise = [0, 0.05, 0.1, 0.15, 0.2]
final_results = pd.DataFrame(columns=['dataset', 'noise', 'averaged_wga', 'single_wga', 'C', 'exp'])
C_VALUES = np.logspace(-4,0,num=20, base=10)

for dataset in datasets:
    # The base path is the directory path of the embeddings (extracted from the base model) of the required datasets. 
    # In the base path, the code expects the embeddings to be in a directory named after the datasets.
    # The code expects the test and validation embeddings along with the test and validation target labels and domain 
    # annotations (The code refers to the domain annotations as groups) in numpy file array format (.npy). For example, 
    # the name of the celebA validation embeddings would be 'celebA_val_embeddings.npy' which is in the 'celebA' directory.

    base_path = '/home/rayyaga2/ThesisWork/ICML24/mixup_exps/datasets/'+dataset+'/'
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
    
    n_features = test_X.shape[1]
    
    for noise_level in noise:
       
       for exp in range(5):
           
            print(dataset, noise_level, exp)
            
            np.random.seed(exp)
    
            test_data = original_val_data.sample(frac=0.5,replace=False)
            train_data = group_sln(original_val_data.drop(test_data.index),p=noise_level)
            full_train_data = pd.concat([group_sln(test_data,p=noise_level), train_data],ignore_index=True)
            
            results = pd.DataFrame(columns=['C', 'val_wga','test_wga','type'])
            
            GUW_best = -np.inf
            for c in C_VALUES:
                _, _, GUW_val, GUW_test = run_exp(train_data,test_data, c)
                results.loc[len(results)] = {'C':c,'alpha':None,'n':None,'val_wga':GUW_val,'test_wga':GUW_test,'type':'GUW'}
              
                
            GUW_avg_param = results[results['type'] == 'GUW'].groupby(['C'])['test_wga'].mean().idxmax()
            
            GUW_wga = np.zeros(10)
            
            GUW_weights = np.zeros((10, n_features))
            GUW_biases = np.zeros(10)
       
    
            for i in range(10):
                weights, bias, _,GUW_wga[i] = run_exp(full_train_data, final_test_data, GUW_avg_param)
                GUW_weights[i] = weights
                GUW_biases[i] = bias[0]
                
            averaged_model = LogisticRegression(penalty='l1', solver='liblinear', C=GUW_avg_param, fit_intercept=True)
            averaged_model.coef_ = GUW_weights.mean(0).reshape(1, n_features)
            averaged_model.intercept_ = np.array([GUW_biases.mean()])
            averaged_model.classes_ = np.unique(test_y)
             
            averaged_wga = wga(averaged_model, final_test_data)
               
            print("Group Upweighted Averged (" + dataset + ")(" + str(noise_level) + "): ", averaged_wga)
            print("Group Upweighted Single (" + dataset + ")(" + str(noise_level) + "): ", GUW_wga[0])
            final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'averaged_wga': averaged_wga, 'single_wga': GUW_wga[0], 'C': GUW_avg_param, 'exp': exp } 
            results = results[0:0] 
            
    path = 'results/averaged_GUW_' + dataset + '.csv'
    final_results.to_csv(path, mode='a', header=True)