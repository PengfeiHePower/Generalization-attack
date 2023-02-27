# remove poison budget, start from a clean pre-trained model, maximize training loss directly
# new objective, inner only consists poison data

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
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--pr', default=0.05, type=float, help='poison rate')
parser.add_argument('--budget', default=50, type=int, help='budget of perturbation size')
parser.add_argument('--sigma', default=0.05, type=float, help='variance of gaussian distribution')
parser.add_argument('--epochs', default=200, type=int, help='num of epochs')
parser.add_argument('--plr', default=0.05, type=float, help='learning rate of poison')
parser.add_argument('--num', default=20, type=int, help='number of gaussian noise')
parser.add_argument('--save', default='p5_lr001', type=str, help='save path for dataloader')
parser.add_argument('--inner', default=10, type=int, help='iteration for inner')
parser.add_argument('--ad', default=0, type = int, help = 'Adaptive lr or not')
#parser.add_argument('')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


start_epoch = 0  # start from epoch 0 or last checkpoint epoch

sharpnessA = []
sharpnessB = []

# prepare datasets
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainsize = len(trainset)

# uniform random initalization of poisons
poisonsize = trainsize
classsize_p = int(poisonsize/10)
poisonimage_np = np.random.uniform(size=(classsize_p*10, 3,32,32))
poisonlabel_np = np.array(list(range(10))*classsize_p)
np.random.shuffle(poisonlabel_np)
poisonsize = classsize_p*10

# load clean data
cleanloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare model, load clean model
print('==> Building model..')
net = ResNet18()
checkpoint = torch.load('./Cifar10checkpoint/ResNet18.pth')
net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
plr_sch = [0.1]*80+[0.01]*40+[0.001]*40+[0.0001]*40+[0.00001]*40


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


print('==> Poisons crafting..')
loss_all = 0
train_n = 0
for batch_id, (images, targets) in enumerate(cleanloader):
    print('batch:', batch_id)
    input_p, target_p = images.to(device), targets.to(device)
    input_p.requires_grad = True
    output_p = net(input_p)
    loss_s = criterion(output_p, target_p)
    loss_s.backward()
    grad = input_p.grad.detach()
    input_p = torch.clamp(input_p + args.plr * torch.sign(grad), min=0.0, max=1.0)
    poisonimage_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = input_p.detach().cpu().numpy()
    poisonlabel_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = target_p.detach().cpu().numpy()
    loss_all = loss_all + loss_s.item()* target_p.size(0)
    train_n = train_n + target_p.size(0)
loss_all = loss_all/train_n
sharpnessB.append(loss_all)
np.savetxt('sharp/test2_sharpB.txt', np.array(sharpnessB))#store sharpness
plt.clf()#visualize sharpness
plt.figure(figsize=(8, 8))
plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
plt.xticks(size=12, weight='bold')
plt.yticks(size=12, weight='bold')
plt.plot(list(range(1,len(sharpnessB)+1)), sharpnessB)
plt.savefig('./figures/test2_sharpB.png')

for epoch in range(args.epochs):
    #load poisons
    poisonset = PoisonTransferCIFAR10Pair(image_np = poisonimage_np, label_np =poisonlabel_np,  train=True, transform=transform_train, download=False)
    poisonloader = torch.utils.data.DataLoader(
        poisonset, batch_size=128, shuffle=False, num_workers=4)

    #inner optimization
    # poison batches
    for _ in range(args.inner):
        train(epoch, net, optimizer, poisonloader)
    
    loss_all = loss_cal(net, criterion, poisonloader, optimizer)
    sharpnessA.append(loss_all)
    np.savetxt('sharp/test2_sharpA.txt', np.array(sharpnessA))#store sharpness
    plt.clf()#visualize sharpness
    plt.figure(figsize=(8, 8))
    plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
    plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.plot(list(range(1,len(sharpnessA)+1)), sharpnessA)
    plt.savefig('./figures/test2_sharpA.png')

    #outer optimization
    loss_all = 0
    train_n = 0
    for batch_id, (images, targets) in enumerate(poisonloader):
        print('batch:', batch_id)
        input_p, target_p = images.to(device), targets.to(device)
        input_p.requires_grad = True
        output_p = net(input_p)
        loss_s = criterion(output_p, target_p)
        loss_s.backward()
        grad = input_p.grad.detach()
        input_p = torch.clamp(input_p + plr_sch[epoch] * torch.sign(grad), min=0.0, max=1.0)
        poisonimage_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = input_p.detach().cpu().numpy()
        poisonlabel_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = target_p.detach().cpu().numpy()
        loss_all = loss_all + loss_s.item()* target_p.size(0)
        train_n = train_n + target_p.size(0)
    loss_all = loss_all/train_n
    sharpnessB.append(loss_all)
    np.savetxt('sharp/test2_sharpB.txt', np.array(sharpnessB))#store sharpness
    plt.clf()#visualize sharpness
    plt.figure(figsize=(8, 8))
    plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
    plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
    plt.xticks(size=12, weight='bold')
    plt.yticks(size=12, weight='bold')
    plt.plot(list(range(1,len(sharpnessB)+1)), sharpnessB)
    plt.savefig('./figures/test2_sharpB.png')


print('==> Data saving..')
np.save('poisoned/test2_gpimage.npy', poisonimage_np)
np.save('poisoned/test2_gplabel.npy', poisonlabel_np)