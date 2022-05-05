#!/bin/sh
#SBATCH -N 1
#SBATCH -p pbatch
#SBATCH -A kdml
#SBATCH -t 04:00:00

echo "Note: Restored Nestrov Momentum!"
cd ..
batch_size=1
ft_batch_size=64
ft_decay=0
ft_epochs=20
ft_learning_rate=$1
ft_momentum=0.9
ft_train_aug=$2 #this is intentional! 
ft_test_aug=test #this is intentional! 
l2sp_weight=-1
learning_rate=-1
lp_decay=-1
momentum=-1
protocol=ft
echo "FT LR: $ft_learning_rate"
echo "FT Train Aug: $ft_train_aug"
python train.py --dataset cifar10 \
        --eval_dataset stl10 \
        --arch resnet50 \
        --protocol $protocol \
        --pretrained_ckpt /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
        --train_aug test\
        --test_aug test \
        --epochs 0\
        --batch_size 64 \
        --num_workers 8 \
        --eval_batch_size 1\
        --learning-rate $learning_rate \
        --momentum $momentum \
        --decay $lp_decay \
        --droprate 0.0 \
        --save ./ckpts \
        --ft_train_aug $ft_train_aug \
        --ft_test_aug $ft_test_aug \
        --ft_epochs $ft_epochs\
        --ft_batch_size $ft_batch_size \
        --ft_learning-rate $ft_learning_rate\
        --ft_momentum $ft_momentum \
        --l2sp_weight $l2sp_weight\
        --ft_decay $ft_decay