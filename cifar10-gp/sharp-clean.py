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
# parser.add_argument('--loaderpath', default='poisoned/resnet18/', type=str, help='path of dataloaders')
parser.add_argument('--name', default='ResNet18', type=str, help='path of models')
parser.add_argument('--gaussian', default=1, type=int, help='sharpness type')
parser.add_argument('--sigma', default=0.05, type=float, help='variance')
args = parser.parse_args()

# class PoisonTransferCIFAR10Pair(CIFAR10):
#     """CIFAR10 Dataset.
#     """
#     def __init__(self, root='data', train=True, transform=None, download=True):
#         super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
#         self.data = (np.load(args.loaderpath+args.name+'_gpimage.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
#         self.targets = np.load(args.loaderpath+args.name+'_gplabel.npy')

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         # print(img[0][0])
#         img = Image.fromarray(img)
#         # print("np.shape(img)", np.shape(img))

#         if self.transform is not None:
#             pos_1 = torch.clamp(self.transform(img), 0, 1)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return pos_1, target


# prepare datasets
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# prepare model
print('==> Building model..')
net = ResNet18()
checkpoint = torch.load('./Cifar10checkpoint/'+args.name+'.pth')
net.load_state_dict(checkpoint['net'])
net = net.to(device)
net.train()
criterion = nn.CrossEntropyLoss()


#compute final sharpness
if args.gaussian == 1:
    sharpness = sharp_cal(net, criterion, trainloader, add_gaussian2, args.sigma)
elif args.gaussian == 0:
    sharpness = sharp_cal(net, criterion, trainloader, add_gaussian, args.sigma)
print('sharpness:', sharpness)