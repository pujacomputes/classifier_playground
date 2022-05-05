#!/bin/sh
#SBATCH -N 1
#SBATCH -p pvis
#SBATCH -A kdml
#SBATCH -t 05:00:00
#SBATCH --job-name="ftnoaugs"

cd ../
PREFIX=/p/lustre1/trivedi1/compnets/classifier_playground/ckpts
#small batch size appears to be needed to handle memory load.
m1_ckpt=$1
m2_ckpt=$2
echo "Model 1: $1"
echo "Model 2: $2"
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