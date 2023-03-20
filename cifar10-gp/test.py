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
parser.add_argument('--loaderpath', default='', type=str, help='loader path')
parser.add_argument('--modelpath', default='poisontest', type=str, help='model path')
parser.add_argument('--savemodel', default='', type=str, help='path of models')
# parser.add_argument('--poisononly', default=False, help='poisononly')
parser.add_argument('--save', default='p5_lr0001', type=str, help='name of dataloaders')
args = parser.parse_args()
print(args)

class PoisonTransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True):
        super(PoisonTransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)
        self.data = (np.load('./poisoned/' + args.loaderpath + '/' + args.save + '_gpimage.npy').transpose([0, 2, 3, 1]) * 255).astype(np.uint8)
        self.targets = np.load('./poisoned/' + args.loaderpath + '/' + args.save + '_gplabel.npy')

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


def test(net, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



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

trainset = torch.utils.data.ConcatDataset([cleanset, poisonset])

poisonloader = torch.utils.data.DataLoader(poisonset, batch_size=128, shuffle=True, num_workers=4)
cleanloader = torch.utils.data.DataLoader(cleanset, batch_size=128, shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)



# prepare model
print('==> Building model..')
net = ResNet18()
checkpoint = torch.load('./Cifar10checkpoint/' + args.modelpath +'/'+args.loaderpath+'/'+args.savemodel+'_RN18_gp.pth')
net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()

print('test performance...')
test(net, testloader)
print('poison train performance...')
test(net, poisonloader)
print('clean train performance...')
test(net, cleanloader)
print('train performance...')
test(net, trainloader)
