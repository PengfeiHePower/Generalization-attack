# remove poison budget, start from a clean pre-trained model, use the smallest lr to update inner model.
# add agumentation to the inner optimization to reduce the gap between poison crafting and retraining.
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
parser.add_argument('--lr_max', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_sch', default = 'fixed', type=str, help='inner lr scheduler')
parser.add_argument('--init', default='rand', type=str, help='init poison model')
parser.add_argument('--pr', default=0.05, type=float, help='poison rate')
# parser.add_argument('--budget', default=50, type=int, help='budget of perturbation size')
parser.add_argument('--sigma', default=0.05, type=float, help='variance of gaussian distribution')
parser.add_argument('--epochs', default=80, type=int, help='num of epochs')
parser.add_argument('--plr', default=0.05, type=float, help='learning rate of poison')
parser.add_argument('--num', default=20, type=int, help='number of gaussian noise')
parser.add_argument('--save', default='p5', type=str, help='save path for dataloader')
parser.add_argument('--inner', default=5, type=int, help='iteration for inner')
parser.add_argument('--plrsch', default='multistep', type = str, help = 'poison lr scheduler')
#parser.add_argument('')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
print(args)


start_epoch = 0  # start from epoch 0 or last checkpoint epoch

sharpness = []

# prepare datasets
print('==> Preparing data..')
transform_poison = transforms.Compose([
    transforms.ToTensor()
])

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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
# load test data
transform_test = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testsize = len(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare model, load clean model
print('==> Building model..')
net = ResNet18()
if args.init == 'pre':
    checkpoint = torch.load('./Cifar10checkpoint/ResNet18.pth')
    net.load_state_dict(checkpoint['net'])
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
# set scheduler type
if args.plrsch == 'fixed':
    plr_sch = [args.plr]*240
elif args.plrsch == 'multistep':
    plr_sch = [0.05]*40+[0.005]*40+[0.0005]*40+[0.00005]*40
    
# lr scheduler
if args.lr_sch == 'fixed':
    def lr_sch(t):
        return args.lr_max
elif args.lr_sch == 'piecewise':
    def lr_sch(t):
        if t / args.inner < 0.7:
            return args.lr_max
        else:
            return args.lr_max / 10.

# training function
def train(epoch, inner_t, net, optimizer, trainloader, lr_sch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        lr = lr_sch(inner_t)
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
    if not os.path.isdir('Cifar10checkpoint/poisongen/resnet18CIA'):
        os.mkdir('Cifar10checkpoint/poisongen/resnet18CIA')
    torch.save(state, './Cifar10checkpoint/poisongen/resnet18CIA/' +args.save +'_train_RN18_gp.pth')
     

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
    
    acc = 100.*correct/total
    # acc_test.append(acc)
    print('Saving..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    if not os.path.isdir('Cifar10checkpoint/poisongen/resnet18CIA'):
        os.mkdir('Cifar10checkpoint/poisongen/resnet18CIA')
    torch.save(state, './Cifar10checkpoint/poisongen/resnet18CIA/' +args.save+'_test_RN18_gp.pth') 


print('==> Poisons crafting..')
for epoch in range(args.epochs):
    #load poisons
    poisonset1 = PoisonTransferCIFAR10Pair(image_np = poisonimage_np, label_np =poisonlabel_np,
                                           train=True, transform=transform_train, download=False)
    poisonset2 = PoisonTransferCIFAR10Pair(image_np = poisonimage_np, label_np =poisonlabel_np,
                                           train=True, transform=transform_poison, download=False)
    poisonloader1 = torch.utils.data.DataLoader(
        poisonset1, batch_size=128, shuffle=False, num_workers=4)
    poisonloader2 = torch.utils.data.DataLoader(
        poisonset2, batch_size=128, shuffle=False, num_workers=4)
    

    #inner optimization
    # poison batches
    innerloader = data_shuffle(poisonloader1, cleanloader, 128) #consider data argumentation in inner
    for inner_t in range(args.inner):
        train(epoch, inner_t, net, optimizer, innerloader, lr_sch)
    test(epoch, net)
    
    # sharp_all = sharp_cal(net, criterion, innerloader, add_gaussian2, args.sigma)
    # sharpness.append(sharp_all)
    # np.savetxt('sharp/resnet18CIA/'+args.save+'_sharp.txt', np.array(sharpness))#store sharpness
    # plt.clf()#visualize sharpness
    # plt.figure(figsize=(8, 8))
    # plt.xlabel('epoch',fontsize=12,color=(0,0,0), weight='bold')
    # plt.ylabel('sharpness',fontsize=12,color=(0,0,0), weight='bold')
    # plt.xticks(size=12, weight='bold')
    # plt.yticks(size=12, weight='bold')
    # plt.plot(list(range(1,len(sharpness)+1)), sharpness)
    # plt.savefig('./figures/resnet18CIA/'+args.save+'_sharp.png')

    #outer optimization
    for batch_id, (images, targets) in enumerate(poisonloader2):
        print('batch:', batch_id)
        input_p, target_p = images.to(device), targets.to(device)
        input_p.requires_grad = True
        loss_grad = 0
        for _ in range(args.num): #estimate expected loss
            net_clone = copy.deepcopy(net).to(device)
            add_gaussian2(net_clone, args.sigma) #add gaussian noise to model parameters
            output_p = net_clone(input_p)
            loss_s = criterion(output_p, target_p)
            loss_s.backward()
            grad = input_p.grad.detach()
            loss_grad = loss_grad+grad
        loss_grad = loss_grad/args.num
        input_p = torch.clamp(input_p + plr_sch[epoch] * torch.sign(loss_grad), min=0.0, max=1.0)
        poisonimage_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = input_p.detach().cpu().numpy()
    
    np.save('poisoned/resnet18CIA/'+args.save+'_gpimage.npy', poisonimage_np)
    np.save('poisoned/resnet18CIA/'+args.save+'_gplabel.npy', poisonlabel_np)


print('==> Data saving..')
np.save('poisoned/resnet18CIA/'+args.save+'_gpimage.npy', poisonimage_np)
np.save('poisoned/resnet18CIA/'+args.save+'_gplabel.npy', poisonlabel_np)