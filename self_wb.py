import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import group_sln, misclassification_self, wga
from sklearn.metrics import accuracy_score
import argparse




noise_level = 0
seed = 0
dataset = 'wb'

np.random.seed(seed)
torch.manual_seed(seed)

def mis_self(val_data, test_data, lr, num_points):
    data = val_data.copy()
    m_self = misclassification_self(data, test_data, base_model_weights, base_model_bias)
    m_self.disagreements(num_points)
    m_self.fit(epochs=500, lr=lr, weight_decay=1e-4, opt='SGD', lr_scheduler = 'step')
    
    return m_self.val_wga(), m_self.test_wga()
    

final_results = pd.DataFrame(columns=['dataset', 'noise', 'wga_mean', 'wga_std', 'lr', 'num_points', 'exp'])


LR_VALUES = [1e-4, 1e-3, 1e-2]
FINETUNE_POINTS = [20, 100, 500]



# The base path is the directory path of the embeddings (extracted from the base model) of the required datasets. 
# In the base path, the code expects the embeddings to be in a directory named after the datasets.
# The code expects the test and validation embeddings along with the test and validation target labels and domain 
# annotations (The code refers to the domain annotations as groups) in numpy file array format (.npy). For example, 
# the name of the celebA validation embeddings would be 'celebA_val_embeddings.npy' which is in the 'celebA' directory.

base_path = './'+dataset+'/'
X = np.load(base_path+dataset+'_val_embeddings.npy')
y = np.load(base_path+dataset+'_val_labels.npy')
group = np.load(base_path+dataset+'_val_groups.npy')
test_X = np.load(base_path+dataset+'_test_embeddings.npy')
test_y = np.load(base_path+dataset+'_test_labels.npy')
test_group = np.load(base_path+dataset+'_test_groups.npy')

original_val_data = pd.DataFrame(X)
original_val_data['target'] = y
original_val_data['group'] = group

final_test_data = pd.DataFrame(test_X)
final_test_data['target'] = test_y
final_test_data['group'] = test_group

print(dataset, noise_level, seed)

# The path is the directory path of the base model trained on the respective datasets. The code expects the base models be present in 
# a directory named 'base_models' with models named after the dataset.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dataset == "multinli" or dataset == "civilcomments":
    path = "./base_models/" + dataset + "_dict.pt"
    state_dict = torch.load(path, map_location=device)
    base_model_weights = state_dict["fc.weight"].cpu().detach().numpy()
    base_model_bias = state_dict["fc.bias"].cpu().detach().numpy()
    
else:
    path = "./base_models/" + dataset + "_model.pt"
    model = torch.load(path, map_location=device)
    base_model_weights = model.fc.weight.cpu().detach().numpy()
    base_model_bias = model.fc.bias.cpu().detach().numpy()
    del model


test_data = original_val_data.sample(frac=0.5,replace=False)

train_data = group_sln(original_val_data.drop(test_data.index).reset_index(drop=True),p=noise_level)

full_train_data = pd.concat([group_sln(test_data.reset_index(drop=True),p=noise_level), train_data],ignore_index=True)


results = pd.DataFrame(columns=['lr', 'num_points', 'val_wga', 'test_wga', 'type'])


for num_points in FINETUNE_POINTS:
    for lr in LR_VALUES:
        rad_val, rad_test = mis_self(train_data, test_data, lr, num_points)
        print(rad_test)
        results.loc[len(results)] = {'lr':lr, 'num_points': num_points, 'val_wga':rad_val,'test_wga':rad_test,'type':'M_SELF'}

avg_param = results[results['type']=='M_SELF'].groupby(['lr', 'num_points'])['test_wga'].mean().idxmax()

print(avg_param)


wgas = np.zeros(10)

seeds = np.random.randint(200, size=(10)) 


for i, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    full_train_data = group_sln(original_val_data.reset_index(drop=True),p=noise_level)
    _,wgas[i] = mis_self(full_train_data, final_test_data, *avg_param)
    print(wgas[i])


print("M_SELF (" + dataset + ")(" + str(noise_level) + "): ", wgas.mean(), wgas.std())
final_results.loc[len(final_results)] = {'dataset': dataset, 'noise': noise_level, 'wga_mean': wgas.mean(), 'wga_std': wgas.std(), 'lr': avg_param[0], 'num_points':avg_param[1], 'exp': seed } 
results = results[0:0] 
            
path = 'results/vanilla_mself/final_mself_' + dataset + '_' + str(noise_level*100) + '.csv'
final_results.to_csv(path, mode='a', header=True)