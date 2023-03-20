import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import copy
from itertools import cycle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_gaussian(model, sigma=1):
    with torch.no_grad():
        for _, param in model.named_parameters():
            std = sigma * torch.std(param, unbiased=False).item()
            param_size = param.size()
            mean_param = torch.zeros(param_size, device=device)
            std_param = std * torch.ones(param_size, device=device)
            gaussian_noise = torch.normal(mean_param, std_param)
            param.add_(gaussian_noise)

def add_gaussian2(model, sigma=0.05):
    with torch.no_grad():
        for _, param in model.named_parameters():
            param_size = param.size()
            mean_param = torch.zeros(param_size, device=device)
            std_param = sigma * torch.ones(param_size, device=device)
            gaussian_noise = torch.normal(mean_param, std_param)
            param.add_(gaussian_noise)

def data_shuffle(poisonloader, cleanloader, batch):
    dataful = []
    for batch_idx, (inputs, targets) in enumerate(poisonloader):
        lens = targets.shape[0]
        for i in range(lens):
            data = (inputs[i], targets[i].item())
            dataful.append(data)

    for batch_idx, (inputs, targets) in enumerate(cleanloader):
        lens = targets.shape[0]
        for i in range(lens):
            data = (inputs[i], targets[i].item())
            dataful.append(data)
    dataloader = torch.utils.data.DataLoader(dataful, batch_size=batch, shuffle=True, num_workers=2)
    return dataloader


def loss_cal(net, criterion, trainloader):
    net.eval()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        net.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        print(batch_idx,':', loss)
        total_loss += loss.detach().item()
    return total_loss/len(trainloader)

def sharp_cal(net, criterion, trainloader, add_gaussian, sigma):
    loss_poison = 0
    train_n = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for _ in range(20):
            net_clone = copy.deepcopy(net)
            add_gaussian(net_clone, sigma)
            output_p = net_clone(inputs)
            loss_s = criterion(output_p, targets)
            loss_poison += loss_s.item() 
            # * targets.size(0)
            # train_n += targets.size(0)
    loss_poison = loss_poison / (20*len(trainloader))
    return loss_poison