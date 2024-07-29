import torch
import torch.optim as optim
import tqdm
import torchvision
from torch.autograd import Variable
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
 
sys.path.insert(1, '/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning')
from group_dataset import imgDataset, get_embed

data = pd.read_csv('celebA_val_data.csv')



trainset = imgDataset(data, split=1, augment_data_flag = False) #splits are as follows: 
 
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


n_classes = trainset.n_classes
model = torch.load('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_high_noise_model.pt')
d = model.fc.in_features

model.fc = torch.nn.Linear(d, n_classes)
model.cuda()
model.eval()
#model.cpu()


train_embeddings_list = []
train_y_list = []
grp_list = []

for batch in tqdm.tqdm(trainloader):
    with torch.no_grad():
        x, y, grp = batch
        embed = get_embed(model, x.cuda()).detach().cpu().numpy() 
        train_embeddings_list.append(embed)
        train_y_list.append(y.detach().cpu().numpy())
        grp_list.append(grp.detach().cpu().numpy())

train_embeddings = np.vstack(train_embeddings_list)
train_y = np.concatenate(train_y_list)
train_grp = np.concatenate(grp_list)

np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_val_embeddings', train_embeddings, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_val_labels', train_y, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_val_groups', train_grp, allow_pickle=False)