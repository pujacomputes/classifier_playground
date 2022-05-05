#!/bin/sh
#SBATCH -N 1
#SBATCH -p pvis
#SBATCH -A kdml
#SBATCH -t 05:00:00
#SBATCH --job-name="1/hua"

cd ..

ft_batch_size=64
ft_decay=0
ft_epochs=20
ft_learning_rate=$1
ft_momentum=0.9
ft_test_aug=test #this is intentional! 
ft_train_aug=$2 #this is intentional! 
resume_lp_ckpt=$3

batch_size=256
epochs=200
learning_rate=30
lp_decay=1e-3
momentum=0.9
protocol=lp+ft
test_aug=test
train_aug=test

echo "LP+FT LR: $ft_learning_rate"
echo "LP+FT Train Aug: $ft_train_aug"
echo "LP+FT Resume Ckpt: $resume_lp_ckpt"
python train.py --dataset cifar10 \
        --eval_dataset stl10 \
        --arch resnet50 \
        --protocol $protocol\
        --pretrained_ckpt /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
        --train_aug $train_aug\
        --test_aug $test_aug\
        --epochs $epochs\
        --batch_size $batch_size \
        --num_workers 8 \
        --eval_batch_size 64\
        --learning-rate $learning_rate \
        --momentum $momentum \
        --decay $lp_decay \
        --droprate 0.0 \
        --save ./ckpts \
        --ft_train_aug $ft_train_aug \
        --ft_test_aug $ft_test_aug \
        --ft_epochs $ft_epochs \
        --ft_learning-rate  $ft_learning_rate\
        --ft_momentum $ft_momentum \
        --resume_lp_ckpt $resume_lp_ckpt \
        --ft_decay $ft_decay 