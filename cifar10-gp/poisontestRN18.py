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
parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--loaderpath', default='resnet18NB', type=str, help='path of dataloaders')
parser.add_argument('--save', default='p5_lr0001', type=str, help='name of dataloaders')
parser.add_argument('--gaussian', default=1, type=int, help='gaussian noise type')
parser.add_argument('--epochs', default=200, type=int, help='training epochs')
parser.add_argument('--poisononly', default=False, help='poisononly')
parser.add_argument('--sharp_cal', default=False, help='calculate sharpness')
parser.add_argument('--saveas', default='', help='save name')
parser.add_argument('--lr_sch', default='piecewise', help='lr scheduler')
parser.add_argument('--opt', default='sgd', help='optimizer')
parser.add_argument('--init', default='rand',type=str, help='initial model')
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

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_acc = 0  # best test accuracy
acc_test = []


# training function
def train(epoch, net, optimizer, trainloader, lr_sch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        lr = lr_sch(epoch + (batch_idx + 1) / len(trainloader))
        optimizer.param_groups[0].update(lr=lr)
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
        
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir('Cifar10checkpoint'):
        os.mkdir('Cifar10checkpoint')
    torch.save(state, './Cifar10checkpoint/poisontest/' + args.loaderpath + '/' +args.saveas+'_train_RN18_gp.pth')
        

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
        torch.save(state, './Cifar10checkpoint/poisontest/' + args.loaderpath + '/' +args.saveas+'_test_RN18_gp.pth')
        best_acc = acc
        


## can change the following module to adopt different datasets
# load clean testing
transform_test = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testsize = len(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

#load poisoned training data
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
poisonset = PoisonTransferCIFAR10Pair(train=True, transform=transform_train, download=False)
cleanset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)

if args.poisononly:
    trainset = poisonset
else:
    trainset = torch.utils.data.ConcatDataset([cleanset, poisonset])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)


# prepare model
print('==> Building model..')
net = ResNet18()
if args.init == 'clean':
    checkpoint = torch.load('./Cifar10checkpoint/ResNet18.pth')
    net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()

### training parameters
# optimizer
if args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
elif args.opt == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_max)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160, 200, 240, 280, 320, 360, 400], gamma=0.1)
# lr scheduler
if args.lr_sch == 'superconverge':
    lr_sch = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_sch == 'piecewise':
    def lr_sch(t):
        if t / args.epochs < 0.4:
            return args.lr_max
        elif t / args.epochs < 0.6:
            return args.lr_max /10.
        elif t / args.epochs < 0.8:
            return args.lr_max /100.
        else:
            return args.lr_max / 1000.
elif args.lr_sch == 'linear':
    lr_sch = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
# elif args.lr_sch == 'onedrop':
#     def lr_sch(t):
#         if t < args.lr_drop_epoch:
#             return args.lr_max
#         else:
#             return args.lr_one_drop
elif args.lr_sch == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_sch == 'cosine':
    def lr_sch(t):
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
elif args.lr_sch == 'cyclic':
    lr_sch = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_sch == 'fixed':
    def lr_sch(t):
        return args.lr_max


for epoch in range(start_epoch, start_epoch+args.epochs): # 200 epochs
    train(epoch, net, optimizer, trainloader, lr_sch)
    test(epoch, net)
    acc_np = np.array(acc_test)
    np.savetxt('results/' + args.loaderpath + '/' + args.saveas +'_testAcc_gp.txt', acc_np)
    # scheduler.step()
    
# compute final sharpness
if args.sharp_cal:
    if args.gaussian == 0:
        sharpness = sharp_cal(net, criterion, trainloader, add_gaussian, 1.0)
    else:
        sharpness = sharp_cal(net, criterion, trainloader, add_gaussian2, 0.05)
    print('sharpness:', sharpness)

plt.figure(figsize=(8, 8))
plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
plt.ylabel('test accuracy',fontsize=12,color=(0,0,0), weight='bold')
plt.xticks(size=12, weight='bold')
plt.yticks(size=12, weight='bold')
plt.plot(list(range(1,len(acc_test)+1)), acc_test)
plt.savefig('./figures/'+ args.loaderpath + '/' + args.saveas+'_acc_gp.png')
print('Figure saved.')
