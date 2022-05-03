#!/bin/sh
#SBATCH -N 1
#SBATCH -p pvis
#SBATCH -A kdml
#SBATCH -t 05:00:00
#SBATCH --job-name="ftnoaugs"

cd ../

#small batch size appears to be needed to handle memory load.
m1_ckpt='/p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar'
m2_ckpt='/p/lustre1/trivedi1/compnets/classifier_playground/ckpts/ft+cifar10_resnet50_ft_base_base_final_checkpoint_020_pth.tar'
m2_ckpt='/p/lustre1/trivedi1/compnets/classifier_playground/ckpts/ft+cifar10_resnet50_ft_test_test_0.03_0.0001_final_checkpoint_020_pth.tar'
#m2_ckpt='none'
python extract_cka.py --dataset_1 cifar10 \
    --dataset_2 none \
    --arch resnet50 \
    --model_1_ckpt $m1_ckpt  \
    --model_2_ckpt $m2_ckpt \
    --dataset_1_trainaug test \
    --dataset_1_testaug test \
    --dataset_2_trainaug test \
    --dataset_2_testaug test \
    --batch_size 32 \
    --eval_batch_size 32 \
    --seed 1