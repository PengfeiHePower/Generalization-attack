# remove poison budget

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from gp import *

from random import sample
import copy
from itertools import cycle



device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='ResNet18 generalization attack')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--pr', default=0.05, type=float, help='poison rate')
parser.add_argument('--budget', default=50, type=int, help='budget of perturbation size')
parser.add_argument('--sigma', default=0.5, type=float, help='variance of gaussian distribution')
parser.add_argument('--epochs', default=200, type=int, help='num of epochs')
parser.add_argument('--plr', default=0.01, type=float, help='learning rate of poison')
parser.add_argument('--num', default=20, type=int, help='number of gaussian noise')
parser.add_argument('--save', default='p5_lr001', type=str, help='save path for dataloader')
#parser.add_argument('')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# epsilon = args.budget/255
sharpness = []

# prepare datasets
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainsize = len(trainset)




# randomly select pr of poisons
poisonsize = int(trainsize * args.pr)
poison_ind = sample(list(range(trainsize)), poisonsize)
clean_ind = [i for i in list(range(trainsize)) if i not in poison_ind]

poisontrain = [trainset[i] for i in poison_ind]
cleantrain = [trainset[i] for i in clean_ind]


poisonloader_o = torch.utils.data.DataLoader(
    poisontrain, batch_size=128, shuffle=True, num_workers=2)

poisonloader_p = copy.deepcopy(poisonloader_o) # preserve for comparison

cleanloader = torch.utils.data.DataLoader(
    cleantrain, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# training function
def train(epoch, net, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# poisonloader initialization
print('==> Poison initialization..')
for batch_id, (image, target) in enumerate(poisonloader_p):
    image, target = image.to(device), target.to(device)
    image = image.uniform_(0, 1)

# pretrain for one epoch
print('==> Pretrain..')
for epoch in range(1):
    innerloader = data_shuffle(poisonloader_o, cleanloader, 128)
    train(epoch, net, optimizer, innerloader)

print('==> Poisons crafting..')


for epoch in range(args.epochs):
    #outer optimization
    sharp_all = 0
    train_n = 0
    for batch_id, (item1, item2) in enumerate(zip(poisonloader_o, cycle(poisonloader_p))):
        print('batch:', batch_id)
        input_p, target_p = item2[0].to(device), item2[1].to(device) #original input
        input_p.requires_grad = True

        loss_grad = 0
        for _ in range(args.num):
            net_clone = copy.deepcopy(net)
            add_gaussian2(net_clone)
            output_p = net_clone(input_p)
            loss_s = criterion(output_p, target_p)
            loss_s.backward()
            grad = input_p.grad.detach()
            loss_grad = loss_grad+grad
            sharp_all += loss_s.detach().item() * target_p.size(0)
            train_n += target_p.size(0)


        loss_grad = loss_grad/args.num
        input_p = torch.clamp(input_p + args.plr * torch.sign(loss_grad), min=0.0, max=1.0)
    
    sharp_all = sharp_all/train_n
    sharpness.append(sharp_all)
    np.savetxt('sharp/resnet18NB2/'+args.save+'_sharp.txt', np.array(sharpness))
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
    plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.plot(list(range(1,len(sharpness)+1)), sharpness)
    plt.savefig('./figures/resnet18NB2/'+args.save+'_sharp.jpg')
    

    #inner optimization
    # poison batches
    print('==> Inner training...')
    innerloader = data_shuffle(poisonloader_p, cleanloader, 128)
    train(epoch, net, optimizer, innerloader)


print('==> Data saving..')
poison_image = []
poison_label = []
for batch_idx, (inputs, targets) in enumerate(poisonloader_p):
    lens = targets.shape[0]
    for i in range(lens):
        poison_image.append(inputs[i].tolist())
        poison_label.append(targets[i].item())
        # input(123)

for batch_idx, (inputs, targets) in enumerate(cleanloader):
    lens = targets.shape[0]
    for i in range(lens):
        poison_image.append(inputs[i].tolist())
        poison_label.append(targets[i].item())

# print(poison_image)
# input(123)
np.save('poisoned/resnet18NB2/'+args.save+'_gpimage.npy', np.array(poison_image))
np.save('poisoned/resnet18NB2/'+args.save+'_gplabel.npy', np.array(poison_label))