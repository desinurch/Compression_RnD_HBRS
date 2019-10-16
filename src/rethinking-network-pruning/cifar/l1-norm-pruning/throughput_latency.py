from __future__ import print_function
import argparse
import numpy as np
import time
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--refine', default='./logs/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to the pruned model to be calculated')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--iterations', default=10, type=int,
                    help='iterations parameter')
parser.add_argument('--dataloc', default='/home/diennur/Documents/RnD/dataset', type=str, metavar='PATH',
                    help='location of dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataloc = args.dataloc

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dataloc, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dataloc, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dataloc, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(dataloc, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.refine:
    checkpoint = torch.load(args.refine, map_location=torch.device('cpu'))
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
	    for data, target in test_loader:
	        if args.cuda:
	            data, target = data.cuda(), target.cuda()
	        # data, target = Variable(data, volatile=True), Variable(target)
	        output = model(data)
	        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
	        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def runThroughput(iterations):
	print('Running prediction...')
	# Allow one warmup prediction outside the overall timing loop
	times = []
	t0 = time.time()
	prec1 = test()
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

# best_prec1 = 0.
# prec1 = test()
iterations = args.iterations
results = runThroughput(iterations)
addResults(iterations, 1, results)