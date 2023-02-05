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

from sam import SAM
from utilitySAM.bypass_bn import enable_running_stats, disable_running_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Poisoned Evaluation')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--loaderpath', default='poisoned/Cifar10/resnet18/united/', type=str, help='path of trainloaders')
parser.add_argument('--name', default='', type=str, help='name of dataloaders')
parser.add_argument('--batch', default=64, type=int, help='batch size')
parser.add_argument('--save', default='', type=str, help='save dir')
args = parser.parse_args()

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_acc = 0  # best test accuracy
acc_test = []


# training function
def train(epoch, net, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        enable_running_stats(net)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad = True)
        
        disable_running_stats(net)
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad = True)

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
        torch.save(state, './Cifar10checkpoint/poisontest/'+args.save+str(args.batch)+'_SGD_ResNet18_gp-sam.pth')
        best_acc = acc

# load clean test
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testsize = len(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#prepare train loader
loaderpath = args.loaderpath
trainloader = torch.load(loaderpath+args.name+str(args.batch)+'_gppoisonedloader.pth')
#input(123)

# prepare model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
base_optimizer = optim.SGD
optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)


for epoch in range(start_epoch, start_epoch+200):
    train(epoch, net, optimizer, trainloader)
    test(epoch, net)
    acc_np = np.array(acc_test)
    np.savetxt('results/poisoned/Cifar10/resnet18/united/'+args.save+str(args.batch)+'_SGD_testAcc_gp-sam.txt', acc_np)
    scheduler.step()
    
#compute final sharpness
loss_poison = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for _ in range(20):
            net_clone = copy.deepcopy(net)
            add_gaussian(net_clone, 1)
            output_p = net_clone(inputs)
            loss_s = criterion(output_p, targets)
            loss_poison = loss_poison + loss_s
    loss_poison = loss_poison.item()/20
print('sharpness:', loss_poison)

plt.figure(figsize=(8, 8))
plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
plt.ylabel('test accuracy',fontsize=12,color=(0,0,0), weight='bold')
plt.xticks(size=12, weight='bold')
plt.yticks(size=12, weight='bold')
plt.plot(list(range(1,len(acc_test)+1)), acc_test)
plt.savefig('./figures/cifar10/resnet18/united/'+args.save+str(args.batch)+'_SGD_acc_gp-sam.jpg')
print('Figure saved.')