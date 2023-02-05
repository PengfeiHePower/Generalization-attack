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
parser.add_argument('--loaderpath', default='poisoned/resnet18/', type=str, help='path of dataloaders')
parser.add_argument('--name', default='', type=str, help='name of dataloaders')
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
        torch.save(state, './Cifar10checkpoint/poisontest/'+args.name+'_RN18_gp.pth')
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
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)


for epoch in range(start_epoch, start_epoch+200): # 200 epochs
    train(epoch, net, optimizer, trainloader)
    test(epoch, net)
    acc_np = np.array(acc_test)
    np.savetxt('results/resnet18/'+args.name+'_testAcc_gp.txt', acc_np)
    scheduler.step()
    
#compute final sharpness
loss_poison = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for _ in range(20):
            net_clone = copy.deepcopy(net)
            add_gaussian(net_clone, 0.5)
            output_p = net_clone(inputs)
            loss_s = criterion(output_p, targets)
            loss_poison = loss_poison + loss_s
    loss_poison = loss_poison.item()/20
print('sharpness:', loss_poison)


loss_poison = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for _ in range(20):
            net_clone = copy.deepcopy(net)
            add_gaussian2(net_clone)
            output_p = net_clone(inputs)
            loss_s = criterion(output_p, targets)
            loss_poison = loss_poison + loss_s
    loss_poison = loss_poison.item()/20
print('sharpness2:', loss_poison)

plt.figure(figsize=(8, 8))
plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
plt.ylabel('test accuracy',fontsize=12,color=(0,0,0), weight='bold')
plt.xticks(size=12, weight='bold')
plt.yticks(size=12, weight='bold')
plt.plot(list(range(1,len(acc_test)+1)), acc_test)
plt.savefig('./figures/resnet18/'+args.name+'_acc_gp.jpg')
print('Figure saved.')