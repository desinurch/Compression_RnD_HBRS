# -*- coding: utf-8 -*-

'''Deep Compression with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning')
parser.add_argument('--loadfile', '-l', default="checkpoint/ckpt.t7",dest='loadfile')
parser.add_argument('--prune', '-p', default=0.5, dest='prune', help='Parameters to be pruned')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--net', default='res50')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()

prune = float(args.prune)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Data
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

kwargs = {'num_workers': 2} if torch.cuda.is_available() else {}
trainset = torchvision.datasets.CIFAR10(root='/home/diennur/Documents/RnD/dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='/home/diennur/Documents/RnD/dataset', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, **kwargs)

# Model
print('==> Building model..')
if args.net=='res50':
    net = ResNet50()
elif args.net=='vgg':
    net = VGG('VGG19')
    
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load weights from checkpoint.
print('==> Loading from checkpoint..')
assert os.path.isfile(args.loadfile), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.loadfile, map_location=torch.device('cpu'))
# net.load_state_dict(checkpoint['net'])
    
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)

# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader.dataset)
    print('Loss: %.4f | Acc: %.3f%% (%d/%d)'
        % (test_loss, 100.*correct/total, correct, total))


def runThroughput(iterations):
    print('Running prediction...')
    # Allow one warmup prediction outside the overall timing loop
    times = []
    t0 = time.time()
    test()
    tn = time.time()
    t = tn - t0
    times.append(t)
    # Overall timing loop for throughput
    t_start = time.time()
    for i in range(iterations):
        # individual timing for latency; we don't support concurrency > 1
        t0 = time.time()
        test()
        tn = time.time()
        t = tn - t0
        times.append(t)
    t_finish = time.time()
    times.insert(0, t_finish - t_start)
    return times

def floatFormat(n):
    return np.format_float_positional(
        n,
        precision=3,
        fractional=False)
    
def addResults(iterations, batch, results):
    # Cast as floats in case they come in as strings
    results = [float(i) for i in results]
    total_sec = np.sum(results[0]);
    ips = (iterations * batch) / float(total_sec)
    # Units of throughput depend on the model
    print("Throughput: %s fps" % (floatFormat(ips)))
    # Since lower is better for latency, extract the max of the 
    # bottom 5%, instead the min of the top 95%
    perc = .95 * 100
    latency_sec = np.percentile(results[1:], perc)
    # Latency is always milliseconds
    ms = latency_sec * 1000
    print("Latency: %s ms" % floatFormat(ms))
    print("Latency StdDev: %f (s)" % np.std(results[1:]))
    return

iterations = 10
results = runThroughput(iterations)
addResults(iterations, 1, results)