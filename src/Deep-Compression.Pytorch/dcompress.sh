#!/bin/bash
#SBATCH --job-name=dcompress
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output weight_compress.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error weight_compress.%N.%j.err  # filename for STDERR


# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pqkd

# locate to your root directory 
cd /home/dnurch2s/Deep-Compression

# run the script
python train_cifar10.py --net res50
python prune.py --prune 0.1
python prune.py --prune 0.25
python prune.py --prune 0.5
python prune.py --prune 0.75