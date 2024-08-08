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
    
np.random.seed(0)
torch.manual_seed(0)
seeds = np.random.randint(100, size=(10))
    
w_decay_sel = 0.001833
epochs_sel = 6
lr_sel = 1e-5
c = 0.007848
lamda = 20

final_results = pd.DataFrame(columns=['dataset', 'noise', 'wga_mean', 'wga_std', 'weight_decay_sel', 'lr_sel', 'epochs_sel', 'c', 'lamda'])

datasets = ['celebA']
noise_level = 0.2

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


    rad_wga = np.zeros(10)
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        test_data = original_val_data.sample(frac=0.5,replace=False)
        train_data = group_sln(original_val_data.drop(test_data.index),p=noise_level)
        full_train_data = pd.concat([group_sln(test_data,p=noise_level), train_data],ignore_index=True)
        
        _,rad_wga[i] = run_misclassify_upweight(full_train_data, final_test_data, c, lamda, epochs_sel, lr_sel, w_decay_sel)
        print(rad_wga[i])
    
    print("RAD (" + dataset + ")(" + str(noise_level) + "): ", rad_wga.mean(), rad_wga.std())
    final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'wga_mean': rad_wga.mean(), 'wga_std': rad_wga.std(), 'weight_decay_sel': w_decay_sel, 'lr_sel': lr_sel, 'epochs_sel': epochs_sel, 'c': c, 'lamda': lamda}
    
    path = 'results/final_RAD_L2_' + dataset + '.csv'
    final_results.to_csv(path, mode='a', header=True)
    