import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter

class NetCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc_in_channels = 32 * 6 * 6
        self.fc1 = nn.Linear(self.fc_in_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_in_channels)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_epoch(train_loader, model, criterion, optimizer, scheduler, 
                epoch, device, log_interval, globaliter, writer):
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    # switch to train mode
    model.train()

    # adjust_learning_rate
    if scheduler is not None:
        scheduler.step()
    
    for batch_idx, (input_data, target) in enumerate(train_loader):
        batch_size = target.size(0)
        
        # TODO: do in other way (this is global batch index, for logging)
        globaliter += 1
      
        # extract batch data
        input_data = input_data.to(device)
        target = target.to(device)

        # compute output
        output = model(input_data)
        loss = criterion(output, target)
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute correct predictions
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        accuracy = 100 * correct / batch_size
        
        losses.update(loss.item(), input_data.size(0))
        accuracies.update(accuracy, input_data.size(0))
        # logging
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(input_data), len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), loss.item()))
        
            # log loss
            writer.add_scalar('Train_Iterations/RunningLoss', loss.item(), globaliter)
            # log LR
            lr = scheduler.get_lr()[0]
            writer.add_scalar('Train_Iterations/LearningRate', lr, globaliter)

    # log the same for epoch
    writer.add_scalar('Train_Epochs/RunningLoss', losses.avg, epoch)
    writer.add_scalar('Train_Epochs/Accuracy', accuracies.avg, epoch)

    return globaliter

def test_epoch(loader, model, criterion, epoch=None, device='cpu', writer=None):
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    # switch to train mode
    model.eval()
    
    with torch.no_grad():
        for (input_data, target) in loader:
            batch_size = target.size(0)
        
            # extract batch data
            input_data = input_data.to(device)
            target = target.to(device)

            # compute output
            output = model(input_data)
            loss = criterion(output, target)

            # compute correct predictions
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = 100 * correct / batch_size
            
            losses.update(loss.item(), input_data.size(0))
            accuracies.update(accuracy, input_data.size(0))

    print('Test Accuracy: {:.2f}%'.format(accuracies.avg))
    if epoch is not None and writer is not None:
        # log the same for epoch
        writer.add_scalar('Test_Epochs/Loss', losses.avg, epoch)
        writer.add_scalar('Test_Epochs/Accuracy', accuracies.avg, epoch)
    else:
        return accuracies.avg



def test_per_class(loader, model, classes, device='cpu'):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))