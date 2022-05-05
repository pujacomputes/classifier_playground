#base, lp
PREFIX=/p/lustre1/trivedi1/compnets/classifier_playground/ckpts
sbatch run_cka.sh /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
    "${PREFIX}/lp+cifar10_resnet50_lp_test_test_200_30.0_0.0_1_-1.0_-1.0_-1_final_checkpoint_200_pth.tar" 
#base, ft
sbatch run_cka.sh /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
    "${PREFIX}/ft+cifar10_resnet50_lp+ft_test_test_200_30.0_0.001_20_3e-05_0.0_-1_final_checkpoint_020_pth.tar"
# #base, ft+lp
sbatch run_cka.sh /p/lustre1/trivedi1/vision_data/moco_v2_800ep_pretrain.pth.tar \
    "${PREFIX}/ft+cifar10_resnet50_lp+ft_test_test_200_30.0_0.001_20_3e-05_0.0_-1_final_checkpoint_020_pth.tar"