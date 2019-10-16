CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python master.py models/mobilenet/prune-by-mac 3 224 224 \
    -im models/mobilenet/model.pth.tar -gp 1 \
    -mi 30 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/ --arch mobilenet