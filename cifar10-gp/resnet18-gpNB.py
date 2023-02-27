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

from PIL import Image
from torchvision.datasets import CIFAR10



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self,  image_np, label_np, root='data',train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root,train=train, download=download, transform=transform)
        self.data = (image_np.transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = label_np

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img[0][0])
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            pos_1 = torch.clamp(self.transform(img), 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, target

parser = argparse.ArgumentParser(description='ResNet18 generalization attack')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--pr', default=0.05, type=float, help='poison rate')
parser.add_argument('--budget', default=50, type=int, help='budget of perturbation size')
parser.add_argument('--sigma', default=1.0, type=float, help='variance of gaussian distribution')
parser.add_argument('--epochs', default=200, type=int, help='num of epochs')
parser.add_argument('--plr', default=0.01, type=float, help='learning rate of poison')
parser.add_argument('--num', default=20, type=int, help='number of gaussian noise')
parser.add_argument('--save', default='p5_lr001', type=str, help='save path for dataloader')
parser.add_argument('--inner', default=10, type=int, help='iteration for inner')
#parser.add_argument('')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


start_epoch = 0  # start from epoch 0 or last checkpoint epoch

sharpness = []

# prepare datasets
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainsize = len(trainset)

# uniform random initalization of poisons
poisonsize = int(trainsize * args.pr)
classsize_p = int(poisonsize/10)
poisonimage_np = np.random.uniform(size=(classsize_p*10, 3,32,32))
poisonlabel_np = np.array(list(range(10))*classsize_p)
np.random.shuffle(poisonlabel_np)
poisonsize = classsize_p*10

# load clean data
cleanloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)


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

def loss_cal(epoch, net, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    total = 0
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total += targets.size(0)
        total_loss += loss.detach().item() * targets.size(0)

    return total_loss, total


print('==> Poisons crafting..')
for epoch in range(args.epochs):
    #load poisons
    poisonset = PoisonTransferCIFAR10Pair(image_np = poisonimage_np, label_np =poisonlabel_np,  train=True, transform=transform_train, download=False)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=128, shuffle=False, num_workers=4)

    #inner optimization
    # poison batches
    innerloader = data_shuffle(poisonloader, cleanloader, 128)
    for _ in range(args.inner):
        train(epoch, net, optimizer, innerloader)

    #outer optimization
    for batch_id, (images, targets) in enumerate(poisonloader):
        print('batch:', batch_id)
        input_p, target_p = images.to(device), targets.to(device)
        input_p.requires_grad = True
        loss_grad = 0
        for _ in range(args.num): #estimate expected loss
            net_clone = copy.deepcopy(net).to(device)
            add_gaussian(net_clone, args.sigma)#add gaussian noise to model parameters
            output_p = net_clone(input_p)
            loss_s = criterion(output_p, target_p)
            loss_s.backward()
            grad = input_p.grad.detach()
            loss_grad = loss_grad+grad
        loss_grad = loss_grad/args.num
        input_p = torch.clamp(input_p + args.plr * torch.sign(loss_grad), min=0.0, max=1.0)
        poisonimage_np[batch_id*128:(min((batch_id+1)*128,poisonsize))]=input_p.detach().cpu().numpy()
    
    # store sharpness
    innerloader = data_shuffle(poisonloader, cleanloader, 128)
    sharp_all = sharp_cal(net, criterion, innerloader, add_gaussian, 1)
    sharpness.append(sharp_all)
    np.savetxt('sharp/resnet18NB/'+args.save+'_sharp.txt', np.array(sharpness))#store sharpness
    plt.clf()#visualize sharpness
    plt.figure(figsize=(8, 8))
    plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
    plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.plot(list(range(1,len(sharpness)+1)), sharpness)
    plt.savefig('./figures/resnet18NB/'+args.save+'_sharp.png')


print('==> Data saving..')
np.save('poisoned/resnet18NB/'+args.save+'_gpimage.npy', poisonimage_np)
np.save('poisoned/resnet18NB/'+args.save+'_gplabel.npy', poisonlabel_np)