#!/bin/bash
#SBATCH --job-name=wprune
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output wprune_PR110.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error wprune_PR110.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/weight-level

# run the script
python cifar.py --dataset cifar10 --arch preresnet --depth 110
python cifar_prune.py --arch preresnet --depth 110 --dataset cifar10 --percent 0.3 --resume ./logs/checkpoint.pth.tar --save_dir ./logs/pruned
python cifar_finetune.py --arch preresnet --depth 110 --dataset cifar10  --resume ./logs/pruned/pruned.pth.tar --save_dir ./logs/finetuned