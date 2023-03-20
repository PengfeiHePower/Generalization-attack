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
parser.add_argument('--modelpath', default='poisontest', type=str, help='model path')
parser.add_argument('--savemodel', default='', type=str, help='path of models')
parser.add_argument('--savepoison', default='', type=str, help='path of data')
parser.add_argument('--gaussian', default=1, type=int, help='gaussian type')
parser.add_argument('--sigma', default = 0.05, type=float, help='variance')
parser.add_argument('--datatp', default='', type=str, help='data type')
parser.add_argument('--sharptype', default='sharp', type=str, help='sharp type')
args = parser.parse_args()
print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.data = (np.load('poisoned/'+args.loaderpath+'/'+args.savepoison+'_gpimage.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load('poisoned/'+args.loaderpath+'/'+args.savepoison+'_gplabel.npy')

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
poisonset = PoisonTransferCIFAR10Pair(train=True, transform=transform_train, download=False)
cleanset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
if args.datatp == 'poison':
    trainloader = torch.utils.data.DataLoader(poisonset, batch_size=128, shuffle=True, num_workers=4)
elif args.datatp == 'clean':
    trainloader = torch.utils.data.DataLoader(cleanset, batch_size=128, shuffle=True, num_workers=4)
elif args.datatp == 'mix':
    trainset = torch.utils.data.ConcatDataset([cleanset, poisonset])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
else:
    print('Invalid data type.')


# prepare model
print('==> Building model..')
net = ResNet18()
checkpoint = torch.load('./Cifar10checkpoint/' + args.modelpath +'/'+args.loaderpath+'/'+args.savemodel+'_RN18_gp.pth')
net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
net.train()


#compute final sharpness
print('==> Computing sharpness..')
if args.sharptype == 'sharp':
    if args.gaussian == 0:
        sharpness = sharp_cal(net, criterion, trainloader, add_gaussian, args.sigma)
    elif args.gaussian == 1:
        sharpness = sharp_cal(net, criterion, trainloader, add_gaussian2, args.sigma)
elif args.sharptype == 'loss':
    sharpness = loss_cal(net, criterion, trainloader)
print('sharpness:', sharpness)