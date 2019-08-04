import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from net import NetCustom
from net import train_epoch, test_epoch, test_per_class

path_data = "./data"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()

use_cuda = not args.no_cuda
batch_size = 64
LR = 0.016
n_epochs = 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = 'cpu' if not (use_cuda and torch.cuda.is_available()) else 'cuda'
print('Using', device)

net = NetCustom().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
trainset = torchvision.datasets.CIFAR10(root=path_data, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=path_data, train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

current_time = "_".join([str(eval("datetime.datetime.now().time().{:0>2}".format(x))) for x in ["hour", "minute", "second"]])
print('Logging into logs/tensorboard/{}'.format(current_time))
path_log = 'logs/tensorboard/' + current_time

writer = SummaryWriter(path_log)

globaliter = 0
for epoch in range(1, n_epochs + 1):
    globaliter = train_epoch(trainloader, net, criterion, optimizer, scheduler, epoch,
                             device, 500, globaliter, writer)
    test_epoch(testloader, net, criterion, epoch, device, writer)