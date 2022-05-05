#!/bin/sh
#SBATCH -N 1
#SBATCH -p pbatch
#SBATCH -A kdml
#SBATCH -t 08:00:00

cd ..

ft_batch_size=1
ft_decay=-1
ft_epochs=1
ft_learning_rate=-1
ft_momentum=-1
ft_test_aug=test #this is intentional! 
ft_train_aug=test #this is intentional! 

batch_size=256
epochs=200
learning_rate=$1
lp_decay=0
momentum=0.9
protocol=lp
test_aug=test
train_aug=$2

echo "LP LR: $learning_rate"
echo "LP Train Aug: $train_aug"
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
        --ft_train_aug test \
        --ft_test_aug test \
        --ft_epochs $ft_epochs \
        --ft_learning-rate  $ft_learning_rate\
        --ft_momentum $ft_epochs \
        --ft_decay $ft_decay 