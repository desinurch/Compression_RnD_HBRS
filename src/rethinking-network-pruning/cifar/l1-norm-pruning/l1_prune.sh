#!/bin/bash
#SBATCH --job-name=l1-prune
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output l1prune_resnet56_cifar10_1.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error l1prune_resnet56_cifar10_1.%N.%j.err  # filename for STDERR


# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/l1_norm_pruning

# run the script
# python main_B.py --dataset cifar10 --arch resnet --depth 56
# python res56prune.py --dataset cifar10 -v A --model ./logs/checkpoint.pth.tar --save ./pruned
python main_finetune.py --refine ./pruned/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56