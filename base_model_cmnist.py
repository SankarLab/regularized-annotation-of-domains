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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_lr = 1e-3
momentum=0.9
weight_decay = 1e-3
epochs = 10
THRESHOLD = 0.5
PATH = 'cmnist_resnet50_new.pt'

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

source_trainloader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ])),
    batch_size=128, shuffle=True, **kwargs)



n_classes = 2
model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
d = model.fc.in_features

model.fc = torch.nn.Linear(d, n_classes)
model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

criterion = torch.nn.CrossEntropyLoss()

print("TRAINING STARTING...\n")

losses = []
batches = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, (inputs, targets, grp) in enumerate(source_trainloader):
    
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        out_probs = torch.sigmoid(outputs)
        #outputs = torch.squeeze(outputs)
        #print(outputs)
        #print(out_probs.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        batches.append(idx+1)
        train_loss += loss.item()
        total += targets.size(0)
        batch_size = targets.shape[0]
        #print(batch_size)

        out_probs = out_probs[:, 1]
        #print(out_probs)
        #print(out_probs.shape)
        out_probs += Variable((torch.ones(batch_size) * (THRESHOLD)).to(device))
        out_probs = torch.floor(out_probs)
        #print(out_probs)
        correct += out_probs.data.eq(targets.data).cpu().sum()
        
        print("TRAIN ACCURACY:", (100.*correct/total).item(), "%")
        print(correct.item(), total)

        print("TRAIN LOSS:", train_loss/(idx+1))
        
print("FINAL TRAINING ACCURACY:", (100.*correct/total).item(), "%")
print("FINAL TRAIN LOSS:", train_loss/(idx+1))

np.save('train_losses.npy', np.array(losses), allow_pickle=False)
np.save('batches.npy', np.array(batches), allow_pickle=False)

torch.save(model, 'cmnist_resnet50_new.pt')