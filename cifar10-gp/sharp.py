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
import copy

from PIL import Image
from torchvision.datasets import CIFAR10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Poisoned Evaluation')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--loaderpath', default='resnet18', type=str, help='path of dataloaders')
parser.add_argument('--name', default='', type=str, help='path of models')
args = parser.parse_args()

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.data = (np.load(args.loaderpath+args.name+'_gpimage.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load(args.loaderpath+args.name+'_gplabel.npy')

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


## can change the following module to adopt different datasets
# load clean testing
transform_test = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testsize = len(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#load poisoned training data
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
trainset = PoisonTransferCIFAR10Pair(train=True, transform=transform_train, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


# prepare model
print('==> Building model..')
net = ResNet18()
checkpoint = torch.load('./Cifar10checkpoint/poisontest/'+args.name+'_RN18_gp.pth')
net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()


#compute final sharpness
for _ in range(5):
    loss_poison = 0
    train_n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for _ in range(20):
                net_clone = copy.deepcopy(net)
                add_gaussian(net_clone, 0.5)
                output_p = net_clone(inputs)
                loss_s = criterion(output_p, targets)
                loss_poison += loss_s.item() * targets.size(0)
            train_n += targets.size(0)
        loss_poison = loss_poison / (train_n * 20)
    print('sharpness1:', loss_poison)


    loss_poison = 0
    train_n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for _ in range(20):
                net_clone = copy.deepcopy(net)
                add_gaussian2(net_clone)
                output_p = net_clone(inputs)
                loss_s = criterion(output_p, targets)
                loss_poison += loss_s.item() * targets.size(0)
            train_n += targets.size(0)
        loss_poison = loss_poison/(train_n * 20)
    print('sharpness2:', loss_poison)