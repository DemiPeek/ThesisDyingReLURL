"""Train CIFAR10 with PyTorch."""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

from models.cifar_model import Cifar10Model
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)

os.environ["WANDB_SILENT"] = "true"
wandb.init(
    project="project_relu_demi",
    config=vars(args),
)
# Define wandb metrics to get nice plots
wandb.define_metric("global_iterations")
# define which metrics will be plotted against it
wandb.define_metric("Train Loss", step_metric="global_iterations")
wandb.define_metric("Train Accuracy", step_metric="global_iterations")
wandb.define_metric("Test Loss", step_metric="global_iterations")
wandb.define_metric("Test Accuracy", step_metric="global_iterations")
wandb.define_metric("Fraction Fully Dead Train", step_metric="global_iterations")
wandb.define_metric("Fraction Weighted Dead Train", step_metric="global_iterations")
wandb.define_metric("Mean Preactivaton Train", step_metric="global_iterations")
wandb.define_metric("Std Preactivaton Train", step_metric="global_iterations")
wandb.define_metric("Ratio negatives Preactivation Train", step_metric="global_iterations")
wandb.define_metric("Fraction Fully Dead Test", step_metric="global_iterations")
wandb.define_metric("Fraction Weighted Dead Test", step_metric="global_iterations")
wandb.define_metric("Mean Preactivaton Test", step_metric="global_iterations")
wandb.define_metric("Std Preactivaton Test", step_metric="global_iterations")
wandb.define_metric("Ratio negatives Preactivation Test", step_metric="global_iterations")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
# CIFAR-10 dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = Cifar10Model()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def calculate_dead_relu(layer):
    is_dead_per_data_point = (layer.abs() < 0.001)
    dead_fractions = is_dead_per_data_point.float().mean(dim=0)
    weighted_count = dead_fractions.sum()
    fully_dead = torch.sum(dead_fractions == 1.0).item()
    weighted_dead = weighted_count.item()
    frac_fully_dead = fully_dead / layer.shape[1]
    frac_weighted_dead = weighted_dead / layer.shape[1]
    return frac_fully_dead, frac_weighted_dead


def layer_metrics(layer):
    array = layer.cpu().detach().numpy()
    mean = np.mean(array)
    std = np.std(array)
    nr_neg = np.sum(array <= 0)
    ratio_neg = nr_neg / np.size(array)
    return mean, std, ratio_neg


# Training
def train(epoch):
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

        train_loss = train_loss/(batch_idx + 1)
        train_acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss, train_acc, correct, total))

    wandb.log({"Train Loss": train_loss,
               "Train Accuracy": train_acc,
               'global_iterations': epoch})

    latent = net.network[:-1](inputs)
    frac_fully_dead, frac_weighted_dead = calculate_dead_relu(latent)
    preactivation = net.network[:-2](inputs)
    mean, std, ratio_neg = layer_metrics(preactivation)

    wandb.log({"Fraction Fully Dead Train": frac_fully_dead,
               "Fraction Weighted Dead Train": frac_weighted_dead,
               "Mean Preactivaton Train": mean,
               "Std Preactivaton Train": std,
               "Ratio negatives Preactivation Train": ratio_neg,
               'global_iterations': epoch})


def test(epoch):
    # global best_acc
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

            test_loss = test_loss/(batch_idx+1)
            test_acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss, test_acc, correct, total))

        wandb.log({"Test Loss": test_loss,
                   "Test Accuracy": test_acc,
                   'global_iterations': epoch})

        latent = net.network[:-1](inputs)
        frac_fully_dead, frac_weighted_dead = calculate_dead_relu(latent)
        preactivation = net.network[:-2](inputs)
        mean, std, ratio_neg = layer_metrics(preactivation)

        wandb.log({"Fraction Fully Dead Test": frac_fully_dead,
                   "Fraction Weighted Dead Test": frac_weighted_dead,
                   "Mean Preactivaton Test": mean,
                   "Std Preactivaton Test": std,
                   "Ratio negatives Preactivation Test": ratio_neg,
                   'global_iterations': epoch})


if __name__ == '__main__':
    # freeze_support()
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
