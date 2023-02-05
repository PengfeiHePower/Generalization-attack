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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_test = []


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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
# clean model
net_clean = ResNet18()
net_clean = net_clean.to(device)
# checkpoint = torch.load('./Cifar10checkpoint/ResNet18.pth')
# net_clean.load_state_dict(checkpoint['net'])


criterion = nn.CrossEntropyLoss()
optimizer_clean = optim.SGD(net_clean.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler_clean = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_clean, T_max=200)
scheduler_clean = torch.optim.lr_scheduler.MultiStepLR(optimizer_clean, milestones=[80, 120, 160], gamma=0.1)

def train(epoch, net, optimizer):
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


def test(epoch, net):
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

    # Save checkpoint.
    acc = 100.*correct/total
    acc_test.append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('Cifar10checkpoint'):
            os.mkdir('Cifar10checkpoint')
        torch.save(state, './Cifar10checkpoint/ResNet18.pth')
        best_acc = acc
    


for epoch in range(start_epoch, start_epoch+200):
    train(epoch, net_clean, optimizer_clean)
    test(epoch, net_clean)
    acc_np = np.array(acc_test)
    np.savetxt('results/resnet18/testAcc_clean.txt', acc_np)
    scheduler_clean.step()

sharp1 = []
sharp2 = []

for _ in range(1):
    loss_poison = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for _ in range(20):
                net_clone = copy.deepcopy(net_clean)
                add_gaussian(net_clone, 0.5)
                output_p = net_clone(inputs)
                loss_s = criterion(output_p, targets)
                loss_poison = loss_poison + loss_s
        loss_poison = loss_poison.item()/20
        sharp1.append(loss_poison)
        print('sharpness:', loss_poison)

    loss_poison = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for _ in range(20):
                net_clone = copy.deepcopy(net_clean)
                add_gaussian2(net_clone)
                output_p = net_clone(inputs)
                loss_s = criterion(output_p, targets)
                loss_poison = loss_poison + loss_s
        loss_poison = loss_poison.item()/20
        sharp2.append(loss_poison)
        print('sharpness2:', loss_poison)

np.savetxt('sharp/resnet18/sharp1_clean.txt', np.array(sharp1))
np.savetext('sharp/resnet18/sharp2_clean.txt', np.array(sharp2))


plt.figure(figsize=(8, 8))
plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
plt.ylabel('test accuracy',fontsize=12,color=(0,0,0), weight='bold')
plt.xticks(size=12, weight='bold')
plt.yticks(size=12, weight='bold')
plt.plot(list(range(1,len(acc_test)+1)), acc_test)
plt.savefig('./figures/resnet18/clean.jpg')
print('Figure saved.')