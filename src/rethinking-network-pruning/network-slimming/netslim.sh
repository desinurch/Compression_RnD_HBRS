#!/bin/bash
#SBATCH --job-name=net-slim
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output netslim_pr56_cifar10.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error netslim_pr56_cifar10.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/network-slimming

# run the script
# python main.py --dataset cifar10 --arch resnet --depth 56
# python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 56 --save ./logs/sparsity_trained
# python resprune.py --dataset cifar10 --depth 56 --percent 0.4 --model ./logs/base/checkpoint.pth.tar --save ./logs/base/pruned
# python resprune.py --dataset cifar10 --depth 56 --percent 0.4 --model ./logs/sparsity_trained/checkpoint.pth.tar --save ./logs/sparsity_trained/pruned
python main_finetune.py --refine ./logs/base/pruned/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56 --save ./logs/base/finetuned
python main_finetune.py --refine ./logs/sparsity_trained/pruned/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56 --save ./logs/sparsity_trained/finetuned