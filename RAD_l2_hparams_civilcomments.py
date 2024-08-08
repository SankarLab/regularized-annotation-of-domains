import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import group_sln, unreg_misclassification_detection, wga


def run_misclassify_upweight(val_data, test_data, c, lam, epochs_sel, lr_sel, w_decay_sel):
    data = val_data.copy()
    detector = unreg_misclassification_detection(val_data, test_data)
    data_ds = detector.run_detection_model(epochs=epochs_sel, lr=lr_sel, weight_decay=w_decay_sel, opt='AdamW', lr_scheduler = 'cosine')
    data = data.drop(data_ds.index)
    full_data = pd.concat([data,data_ds])
    weight = np.concatenate((np.ones(len(data)), lam*np.ones(len(data_ds))))
    model = LogisticRegression(penalty = 'l2', solver='liblinear',C=c).fit(full_data.drop(['target','group'],axis=1), full_data['target'], weight)
    return wga(model,val_data), wga(model, test_data)
    



datasets = ['civilcomments']
noise = [0]
final_results = pd.DataFrame(columns=['dataset', 'noise', 'wga_mean', 'wga_std', 'C', 'lambda', 'iters', 'lr', 'w_decay_sel', 'exp'])
C_VALUES = np.logspace(-4,0,num=20, base=10)
WEIGHT_DECAY_VALUES = np.logspace(-4,0,num=20, base=10)
LR_VALUES = [1e-5]
LAMBDA_VALUES=np.linspace(4, 10, 5)
ITER_VALUES = [6]

for dataset in datasets:

    # The base path is the directory path of the embeddings (extracted from the base model) of the required datasets. 
    # In the base path, the code expects the embeddings to be in a directory named after the datasets.
    # The code expects the test and validation embeddings along with the test and validation target labels and domain 
    # annotations (The code refers to the domain annotations as groups) in numpy file array format (.npy). For example, 
    # the name of the celebA validation embeddings would be 'celebA_val_embeddings.npy' which is in the 'celebA' directory.

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

    for noise_level in noise:

       for exp in range(10):

            print(dataset, noise_level, exp)
            np.random.seed(exp)
            torch.manual_seed(exp)

            test_data = original_val_data.sample(frac=0.5,replace=False)
            train_data = group_sln(original_val_data.drop(test_data.index),p=noise_level)
            full_train_data = pd.concat([group_sln(test_data,p=noise_level), train_data],ignore_index=True)

            results = pd.DataFrame(columns=['C', 'lambda', 'iters', 'lr', 'w_decay_sel', 'val_wga','test_wga','type'])


            for c in C_VALUES:
                for lr_sel in LR_VALUES:
                    for lam in LAMBDA_VALUES:
                        for i in ITER_VALUES:
                            for w_decay_sel in WEIGHT_DECAY_VALUES:
                                print(c,lr_sel,lam,i,w_decay_sel)
                                rad_val, rad_test = run_misclassify_upweight(train_data,test_data, c, lam, i, lr_sel, w_decay_sel)
                                results.loc[len(results)] = {'C':c,'lambda':lam,'iters':i,'lr':lr_sel, 'w_decay_sel': w_decay_sel, 'val_wga':rad_val,'test_wga':rad_test,'type':'RAD'}

            rad_avg_param = results[results['type']=='RAD'].groupby(['C','iters','lr','lambda', 'w_decay_sel'])['test_wga'].mean().idxmax()
           
            print(rad_avg_param)

            rad_wga = np.zeros(10)
            for i in range(10):
                _,rad_wga[i] = run_misclassify_upweight(full_train_data, final_test_data, rad_avg_param[0], rad_avg_param[3], rad_avg_param[1], rad_avg_param[2], rad_avg_param[4])
            
            
            print("RAD (" + dataset + ")(" + str(noise_level) + "): ", rad_wga.mean(), rad_wga.std())
            final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'wga_mean': rad_wga.mean(), 'wga_std': rad_wga.std(), 'C': rad_avg_param[0], 'lambda':rad_avg_param[3], 'iters': rad_avg_param[1], 'lr':rad_avg_param[2], 'w_decay_sel': rad_avg_param[4], 'exp': exp } 
            results = results[0:0] 

final_results.to_csv('results/hparams_RAD_l2_civilcomments.csv', mode='a', header=True)