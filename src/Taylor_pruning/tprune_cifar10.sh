#!/bin/bash
#SBATCH --job-name=tprune-cifar
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output tprune_resnet50_cifar10_1.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error tprune_resnet50_cifar10_1.%N.%j.err  # filename for STDERR


# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/Taylor_pruning

# run the script
python main.py --name=runs/resnet50/resnet50_prune72 --dataset=CIFAR10 --lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=25 --batch-size=256 --pruning=True --seed=0 --model=resnet50 --load_model=./models/pretrained/resnet50-19c8e357.pth --mgpu=False --group_wd_coeff=1e-8 --wd=0.0 --tensorboard=True --pruning-method=22 --no_grad_clip=True --pruning_config=./configs/imagenet_resnet50_prune72.json


