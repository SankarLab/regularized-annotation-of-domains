import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import SGDClassifier, LogisticRegression
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import os
import torch.nn.functional as F



class LogisticRegressionClass(torch.nn.Module):
    def __init__(self, n_features, n_outputs):
        super(LogisticRegressionClass, self).__init__()
        self.linear = torch.nn.Linear(n_features, n_outputs)

    def forward(self, x):
        out = self.linear(x)
        return out
        
def wga(clf, data):
    data = data.copy()
    clf_min = np.inf
    for _, group in data.groupby(['target','group']):
        data = group.drop(['target','group'],axis=1).values
        target = group['target'].values
        clf_min = min(clf.score(data,target),clf_min)
    return clf_min

class EmbeddingsDataset(Dataset):
    def __init__(self, data, weight=False):


        self.y_array = data['target'].to_numpy()
        self.grp_array = data['group'].to_numpy()

        if weight:
            self.weight_array = data['weight'].to_numpy()
            self.in_features = data.drop(['target','group', 'weight'], axis=1).to_numpy()
            self.weight = True
        else:
            self.weight = False
            self.in_features = data.drop(['target','group'], axis=1).to_numpy()

        self.n_classes = np.unique(self.y_array).size
        self.groups = np.unique(self.grp_array).size
        self.n_features = self.in_features.shape[1]

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        grp = self.grp_array[idx]
        x = self.in_features[idx]
        if self.weight:
            w = self.weight_array[idx]
            return x, y, grp, w
        else:
            return x, y, grp


class reg_misclassification_detection:
    def __init__(self, val_data, test_data):

        self.val_df = val_data
        self.test_df = test_data
        self.val_dataset = EmbeddingsDataset(val_data)
        self.test_dataset = EmbeddingsDataset(test_data)

        self.n_outputs = self.val_dataset.n_classes
        self.n_features = self.val_dataset.n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.model = LogisticRegressionClass(self.n_features, self.n_outputs)
        self.model.to(self.device)

    def run_detection_model(self, epochs, lr, weight_decay, c_sel, opt='SGD', lr_scheduler = 'none'):
        
        if opt == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        if (lr_scheduler != 'none') and (lr_scheduler == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif (lr_scheduler != 'none') and (lr_scheduler == 'linear'):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        criterion = torch.nn.CrossEntropyLoss()
        
        trainloader = DataLoader(self.val_dataset, batch_size=32, shuffle=True)
        self.model.train()

        for epoch in range(epochs):
            for i, (x, y, g) in enumerate(trainloader):
                x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss_output = criterion(output, y)
                nweights = 0
                
                reg_loss = torch.tensor(0., requires_grad=True)
                for name , param in self.model.named_parameters():
                    if 'bias' not in name:
                        weights_sum = torch.sum(torch.abs(param))
                        nweights += param.numel()
                        reg_loss = reg_loss + weights_sum
                        
                reg_loss = reg_loss/nweights

                factor = 1/c_sel #lambda
                loss_output += factor * reg_loss
                loss_output.backward()
                optimizer.step()
                
            if lr_scheduler != 'none':
                scheduler.step()

        self.model.eval()
        df = self.val_df.copy()
        val_tensor = torch.tensor(df.drop(['target','group'], axis=1).values).to(self.device)
        preds = torch.argmax(self.model(val_tensor), dim=1).cpu().numpy()
        return df[df['target'] != preds]
    
    
class unreg_misclassification_detection:
    def __init__(self, val_data, test_data):

        self.val_df = val_data
        self.test_df = test_data
        self.val_dataset = EmbeddingsDataset(val_data)
        self.test_dataset = EmbeddingsDataset(test_data)

        self.n_outputs = self.val_dataset.n_classes
        self.n_features = self.val_dataset.n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.model = LogisticRegressionClass(self.n_features, self.n_outputs)
        self.model.to(self.device)

    def run_detection_model(self, epochs, lr, weight_decay, opt='SGD', lr_scheduler = 'none'):
        
        if opt == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        if (lr_scheduler != 'none') and (lr_scheduler == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif (lr_scheduler != 'none') and (lr_scheduler == 'linear'):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

        criterion = torch.nn.CrossEntropyLoss()
        
        trainloader = DataLoader(self.val_dataset, batch_size=32, shuffle=True)
        self.model.train()

        for epoch in range(epochs):
            for i, (x, y, g) in enumerate(trainloader):
                x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss_output = criterion(output, y)
                loss_output.backward()
                optimizer.step()
                
            if lr_scheduler != 'none':
                scheduler.step()

        self.model.eval()
        df = self.val_df.copy()
        val_tensor = torch.tensor(df.drop(['target','group'], axis=1).values).to(self.device)
        preds = torch.argmax(self.model(val_tensor), dim=1).cpu().numpy()
        return df[df['target'] != preds]
        
class misclassification_self:
    def __init__(self, val_data, test_data, base_weights, base_bias):

        self.val_df = val_data
        self.test_df = test_data
        self.val_dataset = EmbeddingsDataset(val_data, noise=True)
        self.test_dataset = EmbeddingsDataset(test_data)

        self.n_outputs = self.val_dataset.n_classes
        self.n_features = self.val_dataset.n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRegressionClass(self.n_features, self.n_outputs)
        self.model.to(self.device)
        self.base_weights = np.copy(base_weights)
        self.base_bias = np.copy(base_bias)
        
        with torch.no_grad():
            self.model.linear.weight = torch.nn.Parameter(torch.from_numpy(self.base_weights).float())
            self.model.linear.bias = torch.nn.Parameter(torch.from_numpy(self.base_bias).float())
            
    def disagreements(self, num_points):
    
        all_orig_logits = []
        all_targets = []
        
        self.model.eval()
        
        trainloader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for i, (x, y, g) in enumerate(trainloader):
                x, y, g = x.to(self.device).float(), y.to(self.device), g.to(self.device)
                
                orig_logits = self.model(x)
 
                all_orig_logits.append(orig_logits)
                all_targets.append(y)
                
        all_orig_logits = torch.cat(all_orig_logits)
        all_targets = torch.cat(all_targets)
        
        if all_targets[0].ndim > 0:
            all_targets = all_targets[:, 0]
        else:
            all_targets = all_targets
            
        loss = F.cross_entropy(all_orig_logits, all_targets, reduction="none").squeeze()
        disagreements = torch.topk(loss, k=num_points)[1].cpu().detach().numpy()
        self.new_val_df = class_balance(self.val_df.iloc[disagreements])
        #print("length of cb finetuning dataset: ", len(self.new_val_df))
        self.new_val_dataset = EmbeddingsDataset(self.new_val_df, noise=True)
        
    

    def fit(self, epochs=50, lr=1e-4, weight_decay=1e-4, opt='AdamW', lr_scheduler = 'none'):

        if opt == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion = torch.nn.CrossEntropyLoss()

        trainloader = DataLoader(self.new_val_dataset, batch_size=32, shuffle=True)
        
        if (lr_scheduler != 'none') and (lr_scheduler == 'cosine'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif (lr_scheduler != 'none') and (lr_scheduler == 'linear'):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        elif (lr_scheduler != 'none') and (lr_scheduler == 'step'):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
        
    
        self.model.train()

        for epoch in range(epochs):
            for i, (x, y, g) in enumerate(trainloader):
                x, y, g = x.to(self.device).float(), y.to(self.device), g.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss_output = criterion(output, y)
                loss_output.backward()
                optimizer.step()
            if lr_scheduler != 'none':
                scheduler.step()

    def test_wga(self):
        data = self.test_df.copy()
        clf_min = np.inf
        self.model.eval()
        for _, group in data.groupby(['target','group']):
            tensor_data = torch.tensor(group.drop(['target','group'],axis=1).values).float().to(self.device)
            preds = torch.argmax(self.model(tensor_data), dim=1).cpu().detach().numpy()
            target = group['target'].values
            clf_min = min(accuracy_score(target, preds),clf_min)
        return clf_min

    def val_wga(self):
        data = self.val_df.copy()
        clf_min = np.inf
        self.model.eval()
        for _, group in data.groupby(['target','group']):
            tensor_data = torch.tensor(group.drop(['target','group', 'true_target'],axis=1).values).float().to(self.device)
            preds = torch.argmax(self.model(tensor_data), dim=1).cpu().detach().numpy()
            target = group['target'].values
            clf_min = min(accuracy_score(target, preds),clf_min)
        return clf_min
  

def group_sln(data:pd.DataFrame, p:float) -> pd.DataFrame:
    noisy = data.copy()
    noisy['group'] = np.abs(noisy['group'] - np.random.choice([0,1], p=[1-p,p], size = noisy['group'].shape))
    return noisy


def class_balance(data:pd.DataFrame) -> pd.DataFrame:
    groups = data.copy().groupby(['target'])
    balanced = groups.apply(lambda df: df.sample(groups.size().min())).reset_index(drop=True)
    return balanced

def group_balance(data:pd.DataFrame) -> pd.DataFrame:
    groups = data.groupby(['target','group'])
    balanced = groups.apply(lambda df: df.sample(groups.size().min())).reset_index(drop=True)
    return balanced


