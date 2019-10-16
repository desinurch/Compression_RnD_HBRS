#!/bin/bash
#SBATCH --job-name=netadapt
#SBATCH --partition=gpu       # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --mem=64GB               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time 1-00:00              # total runtime of job allocation (format D-HH:MM)
#SBATCH --output netadapt_mobilenet_cifar10_1.%N.%j.out # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error netadapt_mobilenet_cifar10_1.%N.%j.err  # filename for STDERR

# load cuda
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/ktb

# locate to your root directory 
cd /home/dnurch2s/netadapt

# run the script
# python train.py ../data/ --dir models/mobilenet/model.pth.tar --arch mobilenet
# python eval.py ../data/ --dir models/mobilenet/model.pth.tar --arch mobilenet
# python build_lookup_table.py --dir latency_lut/lut_mobilenet.pkl --arch mobilenet

#prune by mac
# sh scripts/netadapt_mobilenet-0.5mac.sh
python train.py ../data/ --arch mobilenet --resume models/mobilenet/prune-by-mac/master/iter_5_best_model.pth.tar --dir models/mobilenet/prune-by-mac/master/finetune_model.pth.tar --lr 0.001
python eval.py ../data/ --dir models/mobilenet/prune-by-mac/master/finetune_model.pth.tar --arch mobilenet

#prune by latency
# sh scripts/netadapt_mobilenet-0.5latency.sh
# python train.py ../data/ --arch mobilenet --resume models/mobilenet/prune-by-latency/master/iter_28_best_model.pth.tar --dir models/mobilenet/prune-by-latency/master/finetune_model.pth.tar --lr 0.001
# python eval.py ../data/ --dir models/mobilenet/prune-by-latency/master/finetune_model.pth.tar --arch mobilenet
