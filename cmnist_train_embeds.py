import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from group_dataset import get_embed


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr


class ColoredMNIST(datasets.VisionDataset):
  
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train', 'val', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train, val, test')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target, grp = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target, grp

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'val.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train_set = []
    val_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)
    
      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      if np.random.uniform() < 0.1:
        color_red = not color_red
        
      if color_red:
          red = 1
      else:
          red = 0

      colored_arr = color_grayscale_arr(im_array, red=color_red)
    
      if idx < 30000:
        train_set.append((Image.fromarray(colored_arr), binary_label, red))
      elif idx < 45000:
        val_set.append((Image.fromarray(colored_arr), binary_label, red))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label, red))
    
      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    #dataset_utils.makedir_exist_ok(colored_mnist_dir)
    torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
    torch.save(val_set, os.path.join(colored_mnist_dir, 'val.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))



kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

trainloader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ])),
    batch_size=128, shuffle=True, **kwargs)

n_classes = 2
model = torch.load('/home/rayyaga2/Thesis work/OOD_analysis/mixup_exps/cmnist_resnet50.pt')
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

np.save('/home/rayyaga2/Thesis work/OOD_analysis/mixup_exps/datasets/cmnist/cmnist_train_embeddings', train_embeddings, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/mixup_exps/datasets/cmnist/cmnist_train_labels', train_y, allow_pickle=False)
np.save('/home/rayyaga2/Thesis work/OOD_analysis/mixup_exps/datasets/cmnist/cmnist_train_groups', train_grp, allow_pickle=False)