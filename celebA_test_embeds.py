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
from group_dataset import grpDataset, get_embed, imgDataset

data = pd.read_csv('celebA_test_data.csv')



train_embeddings_list = []
train_y_list = []
grp_list = []


testdata = imgDataset(data, split=2, augment_data_flag = False) #splits are as follows: 
n_classes = testdata.n_classes
testdataloader = DataLoader(testdata, batch_size=64, shuffle=True)

train_embeddings_list = []
train_y_list = []
grp_list = []

model = torch.load('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_model.pt')
d = model.fc.in_features

model.fc = torch.nn.Linear(d, n_classes)
model.cuda()
model.eval()

for batch in tqdm.tqdm(testdataloader):
    with torch.no_grad():
        x, y, grp = batch
        embed = get_embed(model, x.cuda()).detach().cpu().numpy() 
        train_embeddings_list.append(embed)
        train_y_list.append(y.detach().cpu().numpy())
        grp_list.append(grp.detach().cpu().numpy())

train_embeddings = np.vstack(train_embeddings_list)
train_y = np.concatenate(train_y_list)
train_grp = np.concatenate(grp_list)

np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_test_embeddings', train_embeddings, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_test_labels', train_y, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/base_model_tuning/celebA_embeddings/25noise_grp_label_embeddings/celebA_test_groups', train_grp, allow_pickle=False)