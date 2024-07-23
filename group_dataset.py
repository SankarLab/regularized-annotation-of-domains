import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy.typing as t

def get_embed(m, x):
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
def sln(y: t.ArrayLike, p: float) -> t.ArrayLike:
    """Apply symmetric label noise to label array"""
    """Assumes 0/1 label encoding"""
    rng = np.random.default_rng()
    #if p >= 2*np.mean(y):
     #   p = 2*np.mean(y)-0.01
    assert set(y).issubset({0,1}), 'Labels must be in {0,1}'
    assert p < 2*np.mean(y), 'Noise too large for imbalance'
    y = np.array(y)
    n = len(y)
    nflip = int(p*n/2)
    pos_indices = (y>=1).nonzero()[0]
    neg_indices = (y<1).nonzero()[0]
    flip_array_pos=rng.choice(pos_indices, nflip, replace=False)
    flip_array_neg=rng.choice(neg_indices, nflip, replace=False)
    twisted_y = y.copy()
    twisted_y[flip_array_pos] = 0
    twisted_y[flip_array_neg] = 1
    return twisted_y 

def ln(y: t.ArrayLike, p: float) -> t.ArrayLike:
    """Apply label noise to label array which is of only one class"""
    """Assumes 0/1 label encoding"""
    rng = np.random.default_rng()
    assert set(y).issubset({0,1}), 'Labels must be in {0,1}'
    y = np.array(y)
    n = len(y)
    nflip = int(p*n)
    twisted_y = y.copy()
    if 1 in y:
        pos_indices = (y>=1).nonzero()[0]
        flip_array_pos=rng.choice(pos_indices, nflip, replace=False)
        twisted_y[flip_array_pos] = 0
    else:
        neg_indices = (y<1).nonzero()[0]
        flip_array_neg=rng.choice(neg_indices, nflip, replace=False)
        twisted_y[flip_array_neg] = 1
    return twisted_y 


class imgDataset(Dataset):
    def __init__(self, metadata, split=0, augment_data_flag = False, root_dir = '/home/rayyaga2/Thesis work/celebA/data/'): #'/home/rayyaga2/Thesis work/OOD_analysis/fine_tuned_OOD/wb_data/waterbird_complete95_forest2water2' for wb
        
    
        print(len(metadata))
        self.metadata_df = metadata[metadata['split'] == split]
        
        print(len(self.metadata_df))
        self.transform = get_transform_cub((224, 224), augment_data = augment_data_flag)
        self.y_array = self.metadata_df['y'].values
        self.grp_array = self.metadata_df['place'].values
        self.root_dir = root_dir
      
        self.n_classes = np.unique(self.y_array).size
        
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        grp = self.grp_array[idx]
        

        img_path = os.path.join(
                self.root_dir,
                self.filename_array[idx])
        
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y, grp
        
class noisyDataset(Dataset):
    def __init__(self, metadata, split=0, noise_level=0.15, augment_data_flag = False, root_dir = '/home/rayyaga2/Thesis work/celebA/data/'): #'/home/rayyaga2/Thesis work/OOD_analysis/fine_tuned_OOD/wb_data/waterbird_complete95_forest2water2' for wb
        
    
        print(len(metadata))
        self.metadata_df = metadata[metadata['split'] == split]
        
        print(len(self.metadata_df))
        self.transform = get_transform_cub((224, 224), augment_data = augment_data_flag)
        self.y_array = sln(self.metadata_df['y'].values, p = noise_level)
        self.root_dir = root_dir
      
        self.n_classes = np.unique(self.y_array).size
        
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        

        img_path = os.path.join(
                self.root_dir,
                self.filename_array[idx])
        
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y
        
class grpDataset(Dataset):
    def __init__(self, metadata, split=2, group=0, augment_data_flag = False, root_dir = '/home/rayyaga2/Thesis work/celebA/data/'):
        
    
        print(len(metadata))
        self.metadata_df = metadata[(metadata['split'] == split) & (metadata['group'] == group)]
        print(len(self.metadata_df))
        self.transform = get_transform_cub((224, 224), augment_data = augment_data_flag)
        self.y_array = self.metadata_df['y'].values
        self.grp_array = self.metadata_df['place'].values
        self.root_dir = root_dir
      
        self.n_classes = np.unique(self.y_array).size
        
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        grp = self.grp_array[idx]
        

        img_path = os.path.join(
                self.root_dir,
                self.filename_array[idx])
        
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y, grp
        
class grp_noiseDataset(Dataset):
    def __init__(self, metadata, split=0, minority_noise_level=0.25, majority_noise_level=0.1, augment_data_flag = False, root_dir = '/home/rayyaga2/Thesis work/celebA/data/'): #'/home/rayyaga2/Thesis work/OOD_analysis/fine_tuned_OOD/wb_data/waterbird_complete95_forest2water2' for wb
        
    
        print(len(metadata))
        self.metadata_df = metadata[metadata['split'] == split]
        grp_sizes = []
        for grp in np.unique(self.metadata_df['group'].values):
            grp_sizes.append(len(self.metadata_df[self.metadata_df['group'] == grp]))
        minority_grp = np.argmin(np.array(grp_sizes))
        print(minority_grp)
        
        self.metadata_df.loc[self.metadata_df['group'] == minority_grp, 'y'] =  ln(self.metadata_df[self.metadata_df['group'] == minority_grp]['y'].values, p = minority_noise_level)
        self.metadata_df.loc[self.metadata_df['group'] != minority_grp, 'y'] =  sln(self.metadata_df[self.metadata_df['group'] != minority_grp]['y'].values, p = majority_noise_level)
        print(len(self.metadata_df))
        self.transform = get_transform_cub((224, 224), augment_data = augment_data_flag)
        self.y_array = self.metadata_df['y'].values
        self.root_dir = root_dir
      
        self.n_classes = np.unique(self.y_array).size
        
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        

        img_path = os.path.join(
                self.root_dir,
                self.filename_array[idx])
        
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return img, y


    
def get_transform_cub(target_resolution, augment_data):
    scale = 256.0 / 224.0

    if (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def get_transform_celeba(target_resolution):

    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    transform = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform


def get_loader(data, train, reweight_groups, reweight_classes, reweight_places, **kwargs):
    if not train: # Validation or testing
        assert reweight_groups is None
        assert reweight_classes is None
        assert reweight_places is None
        shuffle = False
        sampler = None
    elif not (reweight_groups or reweight_classes or reweight_places): # Training but not reweighting
        shuffle = True
        sampler = None
    elif reweight_groups:
        # Training and reweighting groups
        # reweighting changes the loss function from the normal ERM (average loss over each training example)
        # to a reweighted ERM (weighted average where each (y,c) group has equal weight)
        group_weights = len(data) / data.group_counts
        weights = group_weights[data.group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    elif reweight_classes:  # Training and reweighting classes
        class_weights = len(data) / data.y_counts
        weights = class_weights[data.y_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False
    else: # Training and reweighting places
        place_weights = len(data) / data.p_counts
        weights = place_weights[data.p_array]
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)
        shuffle = False

    loader = DataLoader(
        data,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def log_data(logger, train_data, test_data, val_data=None, get_yp_func=None):
    logger.write(f'Training Data (total {len(train_data)})\n')
    # group_id = y_id * n_places + place_id
    # y_id = group_id // n_places
    # place_id = group_id % n_places
    for group_idx in range(train_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {train_data.group_counts[group_idx]:.0f}\n')
    logger.write(f'Test Data (total {len(test_data)})\n')
    for group_idx in range(test_data.n_groups):
        y_idx, p_idx = get_yp_func(group_idx)
        logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {test_data.group_counts[group_idx]:.0f}\n')
    if val_data is not None:
        logger.write(f'Validation Data (total {len(val_data)})\n')
        for group_idx in range(val_data.n_groups):
            y_idx, p_idx = get_yp_func(group_idx)
            logger.write(f'    Group {group_idx} (y={y_idx}, p={p_idx}): n = {val_data.group_counts[group_idx]:.0f}\n')
