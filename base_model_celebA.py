import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
from group_dataset import imgDataset
from torch.utils.data import DataLoader
import pandas as pd

USE_CUDA = torch.cuda.is_available()

init_lr = 1e-3
momentum=0.9
weight_decay = 1e-4
epochs = 50
THRESHOLD = 0.5
PATH = 'celebA_train_resnet50.pt'

data = pd.read_csv('celebA_group_data.csv')



source_trainset = imgDataset(data, split=0, augment_data_flag = True) #splits are as follows: 
#source_testset = CelebADataset(data, split=1)  # 0 -> source training, 1 -> source testing, 2 -> target training, 3 -> target testing

source_trainloader = DataLoader(source_trainset, batch_size=128, shuffle=True)
#source_testloader = DataLoader(source_testset, batch_size=32, shuffle=True)

n_classes = source_trainset.n_classes
model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
d = model.fc.in_features

model.fc = torch.nn.Linear(d, n_classes)
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

criterion = torch.nn.CrossEntropyLoss()

print("TRAINING STARTING...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, (inputs, targets) in enumerate(source_trainloader):
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

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
        train_loss += loss.item()
        total += targets.size(0)
        batch_size = targets.shape[0]
        #print(batch_size)

        out_probs = out_probs[:, 1]
        #print(out_probs)
        #print(out_probs.shape)
        out_probs += Variable((torch.ones(batch_size) * (THRESHOLD)).cuda())
        out_probs = torch.floor(out_probs)
        #print(out_probs)
        correct += out_probs.data.eq(targets.data).cpu().sum()
        
        print("TRAIN ACCURACY:", (100.*correct/total).item(), "%")
        print(correct.item(), total)

        print("TRAIN LOSS:", train_loss/(idx+1))
        
print("FINAL TRAINING ACCURACY:", (100.*correct/total).item(), "%")
print("FINAL TRAIN LOSS:", train_loss/(idx+1))

torch.save(model, 'celebA_train_entire_model_50epochs.pt')