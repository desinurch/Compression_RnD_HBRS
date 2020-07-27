from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import models.cifar as models
from utils.misc import get_conv_zero_param
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[41, 42],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='./pruned', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=56, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=1, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save_dir', default='finetuned/', type=str)
#Device options
parser.add_argument('--percent', default=0.6, type=float)
parser.add_argument('--iterations', default=1024, type=int,
                    help='iterations parameter')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    testset = dataloader(root='../data', train=False, download=False, transform=transform_test)
    rand_sampler = torch.utils.data.RandomSampler(testset, num_samples=1, replacement=True)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, sampler=rand_sampler, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('preresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = False
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) # default is 0.001

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(checkpoint['state_dict'])

    # logger = Logger(os.path.join(args.save_dir, 'log_finetune.txt'), title=title)
    # logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    epoch = start_epoch
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
    num_parameters = get_conv_zero_param(model)
    print('Zero parameters: {}'.format(num_parameters))
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print('Parameters: {}'.format(num_parameters))

    print('Best acc:')
    print(best_acc)

    iterations = args.iterations
    # results = runThroughput(iterations, testloader, model, criterion, epoch, use_cuda)
    results = runThroughput(iterations, testloader, model)
    addResults(iterations, 1, results)


def test_old(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time()
    # bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        # data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    print('Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(loss=losses.avg, top1=top1.avg, top5=top5.avg,))
    #     # measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=len(testloader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 top1=top1.avg,
    #                 top5=top5.avg,
    #                 )
    #     bar.next()
    # bar.finish()
    return (losses.avg, top1.avg)

def test(test_loader, model, max=5, warmup=0, iterations=1):
    model.eval()
    correct = 0
    #Warmup
    for i in range(warmup):
      iterator = iter(test_loader)
      data, target = iterator.next()
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #Measure
    times = []
    for i in range(iterations):
      t0 = time.time()
      iterator = iter(test_loader)
      data, target = iterator.next()
      output = model(data)
      pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
      correct += pred.eq(target.data.view_as(pred)).cpu().sum()      
      t1 = time.time()
      ts = t1 - t0
      times.append(ts)
    
    #Report
    results = {"seconds":times, "predictions":correct}
    return results

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']

def runThroughput(iterations, testloader, model):
    print('Running prediction...')
    # Allow one warmup prediction outside the overall timing loop
    times = []
    t0 = time.time()
    # test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
    test(testloader,model)
    tn = time.time()
    t = tn - t0
    times.append(t)
    # Overall timing loop for throughput
    t_start = time.time()
    for i in range(iterations):
        # individual timing for latency; we don't support concurrency > 1
        t0 = time.time()
        # test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        test(testloader,model)
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

if __name__ == '__main__':
    main()
