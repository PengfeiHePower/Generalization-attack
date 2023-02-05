import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse


parser = argparse.ArgumentParser(description='training data united')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--loaderpath', default='poisoned/Cifar10/resnet18/', type=str, help='path of dataloaders')
parser.add_argument('--name', default='p5b8R11_', type=str, help='name of dataloaders')
parser.add_argument('--save', default='poisoned/Cifar10/resnet18/united/', type=str, help='save dir')
args = parser.parse_args()

loaderpath = args.loaderpath
poisonloader = torch.load(loaderpath+args.name+'gppoisonedloader.pth')
cleanloader = torch.load(loaderpath+args.name+'cleanloader.pth')

dataful = []
for batch_idx, (inputs, targets) in enumerate(poisonloader):
    lens = targets.shape[0]
    for i in range(lens):
        data = (inputs[i], targets[i].item())
        dataful.append(data)
        # input(123)

for batch_idx, (inputs, targets) in enumerate(cleanloader):
    lens = targets.shape[0]
    for i in range(lens):
        data = (inputs[i], targets[i].item())
        dataful.append(data)

#dataloader = torch.utils.data.DataLoader(dataful, batch_size=args.batch, shuffle=True, num_workers=2)

print('==> Data saving..')
torch.save(dataful, args.save+args.name+'_gppoisoned.pt')
#torch.save(dataloader, args.save+args.name+str(args.batch)+'_gppoisonedloader.pth')