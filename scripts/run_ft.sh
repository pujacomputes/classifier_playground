#!/bin/sh
#SBATCH -N 1
#SBATCH -p pvis
#SBATCH -A kdml
#SBATCH -t 05:00:00
#SBATCH --job-name="base"

cd ..
train_aug=base
batch_size=256
lp_decay=1e-4
protocol=ft
ft_decay=0
python train.py --dataset cifar10 \
        --eval_dataset stl10 \
        --arch resnet50 \
        --protocol $protocol \
        --pretrained_ckpt /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
        --train_aug $train_aug\
        --test_aug test \
        --epochs 200\
        --batch_size $batch_size \
        --num_workers 8 \
        --eval_batch_size $batch_size\
        --learning-rate 0.03 \
        --momentum 0.9 \
        --decay $lp_decay \
        --droprate 0.0 \
        --save ./ckpts \
        --ft_train_aug base \
        --ft_test_aug test \
        --ft_epochs 20 \
        --ft_learning-rate 0.03 \
        --ft_momentum 0.9 \
        --ft_decay $ft_decay 