import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import copy
from itertools import cycle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_gaussian(model, sigma):
    with torch.no_grad():
        for _, param in model.named_parameters():
            std = sigma * torch.std(param, unbiased=False).item()
            param_size = param.size()
            mean_param = torch.zeros(param_size)
            std_param = std * torch.ones(param_size)
            gaussian_noise = torch.normal(mean_param, std_param).to(device)
            param.add_(gaussian_noise)

def add_gaussian2(model, sigma=0.05):
    with torch.no_grad():
        for _, param in model.named_parameters():
            param_size = param.size()
            mean_param = torch.zeros(param_size)
            std_param = sigma * torch.ones(param_size)
            gaussian_noise = torch.normal(mean_param, std_param).to(device)
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

def add_perturbation(perturbation, poisonloader):
    poisonloader_p = copy.deepcopy(poisonloader)
    perturbation_p = copy.deepcopy(perturbation)
    for batch_id, (item1, item2) in enumerate(zip(poisonloader_p, cycle(perturbation_p))):
        item1[0] = item1[0]+item2[0]
    return poisonloader_p