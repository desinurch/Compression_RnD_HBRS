# Research and Development Project
## A Comparative Study of Sparsity Methods in Deep Neural Network for Faster Inference

Code and documentation for research and development project with a topic in Deep Neural Network Compression as partial fulfillment in Masters of Autonomous Systems program.

## Overview

Comparison of compression methods in Deep Learning for image classification task. Comparison is done in terms of speed using the backbone of [MLMark benchmark](https://www.eembc.org/mlmark/). Compression methods observed are as follows:

![Compression Methods](/imgs/methods-compression.png)

## Description
### Dataset
Dataset used for comparison is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) to mimic real-life situations.

### Model Architecture
Dataset are processed using the network of ResNet-56 and ResNet-110 with pre-activations. In model distillation mode, both of the network act as a teacher which knowledge are transferred to student networks; ResNet-1, ResNet-10, and ResNet-20

## Results
![Speedup vs Compression](/imgs/speedup_vs_compression.png)
 ---------------------------------------------------------------------------
![Accuracy vs Speedup](/imgs/accuracyLoss_vs_sppedUp.png)

### Repository structure:
- L1 norm pruning: cloned from https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/l1-norm-pruning with minor modifications. Based on the implementation of the paper [Pruning Filters For Efficient ConvNets](https://arxiv.org/pdf/1608.08710.pdf)
- Weight level pruning : cloned from https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/weight-level with minor modifications. Based on the implementation of the paper [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)
- Knowledge Distillation methods : 
	- Cloned from https://github.com/peterliht/knowledge-distillation-pytorch with minor modifications. Based on the implementation of the paper [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)
	- FitNets implementation. Cloned from https://github.com/AberHu/Knowledge-Distillation-Zoo with modifications. Based on the implementation of the paper [FitNets: Hints for Thin Deep Nets](https://arxiv.org/pdf/1412.6550.pdf)
- Low Rank Approximations [Caffe: Convolutional Architecture for Fast Feature Embedding](https://arxiv.org/abs/1408.5093)
- Quantizations (https://github.com/eladhoffer/convNet.pytorch/blob/master/models/modules/quantize.py)
- Results presentations
