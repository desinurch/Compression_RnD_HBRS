#!/bin/bash
#SBATCH --job-name=fitnet
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 0-12:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output fitnet_PR110/R20_cifar10.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error fitnet_PR110/R20_cifar10.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/FitNets

# run the script
python train_fitnet_new.py --s_init=logs/checkpoint/baseline_r20_000.pth.tar \
	    --t_model=logs/checkpoint/baseline_pr110.pth.tar \
		--data_name=cifar10  \
		--t_name=preresnet110 \
		--s_name=resnet20 \
		--num_class=10