# remove poison budget, start from a clean pre-trained model, 
# inner only consists of poison data, add penality to reduce the discrepency between poison and clean loss
# restart the inner after epochs to fit different local minimizers
# apply methods in metapoison, ensembing and reinitialize

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
parser.add_argument('--init', default='rand', type=str, help='init poison model')
parser.add_argument('--tlr', default=0.1, type=float, help='retrain learning rate(inner)')
parser.add_argument('--pr', default=0.05, type=float, help='poison rate')
# parser.add_argument('--budget', default=50, type=int, help='budget of perturbation size')
parser.add_argument('--sigma', default=0.05, type=float, help='variance of gaussian distribution')
parser.add_argument('--epochs', default=60, type=int, help='num of generation epochs')
parser.add_argument('--plr', default=200, type=float, help='max learning rate of poison(outer)')
parser.add_argument('--num', default=20, type=int, help='number of gaussian noise')
parser.add_argument('--save', default='p5-lam5', type=str, help='save path for dataloader')
parser.add_argument('--inner', default=2, type=int, help='iteration for inner unroll')
parser.add_argument('--tlrsch', default = 'multistep', type = str, help = 'retrain lt schedule')
parser.add_argument('--plrsch', default='multistep', type = str, help = 'poison lr scheduler')
parser.add_argument('--lam', default=5, type=float, help='penalty coefficient')
parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
parser.add_argument('--nummodel', default=24, type=int, help='num of models')
parser.add_argument('--T', default=200, type=int, help='retraining epochs')
#parser.add_argument('')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


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


## prepare model, load checkpoints, store models and training epochs into a dict
print('==> Loading models..')
model_dict = {}
epoch_dict = {}
for i in range(1,args.nummodel+1):
    checkpoint_id = math.floor(i*args.T/args.nummodel)
    net = ResNet18()
    checkpoint = torch.load('./Cifar10checkpoint/clean/ResNet18-init_'+str(i)+'.pth')
    net.load_state_dict(checkpoint['net'])
    model_dict[i] = net
    epoch_dict[i] = checkpoint_id

criterion = nn.CrossEntropyLoss()
# if args.opt == 'sgd':
#     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# elif args.opt == 'adam':
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# set scheduler for plr
if args.plrsch == 'fixed':
    def plr_sch(t):
        return args.plr
elif args.plrsch == 'multistep':
    def plr_sch(t):
        plr_list = [200]*20+[20]*20+[2]*20
        return plr_list[t]
    
if args.tlrsch == 'fixed':
    def tlr_sch(t):
        return args.tlr
elif args.tlrsch == 'multistep':
    def tlr_sch(t):
        tlr_list = [0.1]*80+[0.01]*40+[0.001]*40+[0.0001]*40
        return tlr_list[t]

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
        # lr = lr_schedule(epoch)
        # opt.param_groups[0].update(lr=lr)
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
    # state = {
    #     'net': net.state_dict(),
    #     'epoch': epoch,
    # }
    # if not os.path.isdir('Cifar10checkpoint/poisongen/resnet18PPMR'):
    #     os.mkdir('Cifar10checkpoint/poisongen/resnet18PPMR')
    # torch.save(state, './Cifar10checkpoint/poisongen/resnet18PPMR/' +args.save +'_train_RN18_gp.pth')
     

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
    # print('Saving..')
    # state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    # }
    # if not os.path.isdir('Cifar10checkpoint/poisongen/resnet18PPMR'):
    #     os.mkdir('Cifar10checkpoint/poisongen/resnet18PPMR')
    # torch.save(state, './Cifar10checkpoint/poisongen/resnet18PPMR/' +args.save+'_test_RN18_gp.pth') 



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
    
    innerloader = data_shuffle(poisonloader1, cleanloader, 128)
    
    model_proxy_list={}
    for i in range(1, args.nummodel): # unroll model 
        net_proxy = copy.deepcopy(model_dict[i]).to(device) ## unroll copy
        optimizer_proxy = optim.SGD(net_proxy.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for _ in range(args.inner): ### unroll for each model
            train(epoch, net_proxy, optimizer_proxy, innerloader)
        test(epoch, net_proxy)
        model_proxy_list[i] = net_proxy.cpu()
        
    for batch_id, (images, targets) in enumerate(poisonloader2): # compute adv loss and update poison
        print('batch:', batch_id)
        input_p, target_p = images.to(device), targets.to(device)
        input_p.requires_grad = True
        grad_proxy = 0
        for i in range(1, args.nummodel):
            net_proxy = model_proxy_list[i].to(device)
            loss_sharp = 0
            for _ in range(args.num): #estimate expected loss
                net_clone = copy.deepcopy(net_proxy).to(device)
                add_gaussian2(net_clone, args.sigma) #add gaussian noise to model parameters
                output_p = net_clone(input_p)
                loss_s = criterion(output_p, target_p)
                # loss_s.backward()
                # grad = input_p.grad.detach()
                # loss_grad = loss_grad+grad
                loss_sharp += loss_s
            # loss_grad = loss_grad/args.num
            loss_sharp = loss_sharp/args.num #poison sharpness
            loss_true = criterion(net_proxy(input_p), target_p) #poison nature loss
            print('poison loss:', loss_true)
            print('poison sharpness:', loss_sharp)
            # loss_true.backward()
            loss_total = loss_sharp - args.lam * loss_true #composite loss function
            loss_total.backward()
            grad = input_p.grad.detach()
            grad_proxy += grad
        grad_proxy = grad_proxy / args.nummodel
        ## update poison samples
        input_p = torch.clamp(input_p + plr_sch(epoch)/args.T * torch.sign(grad_proxy), min=0.0, max=1.0)
        poisonimage_np[batch_id*128:(min((batch_id+1)*128,poisonsize))] = input_p.detach().cpu().numpy()
        np.save('poisoned/resnet18PPMR/'+args.save+'_gpimage.npy', poisonimage_np)
        np.save('poisoned/resnet18PPMR/'+args.save+'_gplabel.npy', poisonlabel_np)
    
    ## roll models for 1 step
    for i in range(1, args.nummodel):
        # print('roll:',i)
        epoch_roll = epoch_dict[i]
        # print('epoch_roll:', epoch_roll)
        if epoch_roll == args.T+1:
            epoch_roll = 1
            model_roll = ResNet18()
            model_dict[i] = model_roll
            epoch_dict[i] = epoch_roll
        else:
            model_roll = model_dict[i].to(device)
            optimizer_roll = optim.SGD(model_roll.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            lr = tlr_sch(epoch_roll-1)
            optimizer_roll.param_groups[0].update(lr=lr)
            train(epoch_roll-1, model_roll, optimizer_roll, cleanloader)
            epoch_roll += 1
            model_dict[i] = model_roll.cpu()
            epoch_dict[i] = epoch_roll
        

print('==> Data saving..')
np.save('poisoned/resnet18PPMR/'+args.save+'_gpimage.npy', poisonimage_np)
np.save('poisoned/resnet18PPMR/'+args.save+'_gplabel.npy', poisonlabel_np)