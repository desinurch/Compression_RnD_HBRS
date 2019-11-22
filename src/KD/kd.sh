#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output kd_PR110+R56_cifar10.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error kd_PR110+R56_cifar10.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/KD

# run the script
# python train.py --model_dir experiments/base_resnet56 --restore_file best
python train.py --model_dir experiments/resnet18_distill/preresnet_teacher
python train.py --model_dir experiments/resnet18_distill/resnet_teacher
# python train.py --model_dir experiments/cnn_distill
