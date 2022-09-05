## Instructions

```
pip install torch==1.11.0
mkdir logs
mkdir ckpts
mkdir clip_ckpts
mkdir clip_safety_logs
cd clip_scripts
./run_ft.sh
```

- ```train.py``` is the main file, which accepts command line arguments to run different protocols (lp, ft, lp+ft) using different augmentation strategies. Note: to replicate the feature distortion paper, the ```train_aug``` and ```ft_train_aug``` should be set to ```test```. Doing so, prevents horizontal flipping and cropping from being used as augmentations.  
- ```train_clip.py``` contains support for running ```domainnet``` using a pretrained ```CLIP``` model.
- ```train_vat_clip.py``` contains support for running ```vat-lp``` on  ```domainnet``` using a pretrained ```CLIP``` model.
- ```safety_eval_clip.py``` contains support for evaluation of ``` on  ```domainnet``` using a pretrained ```CLIP``` model.
Please note, in future releases files supporting ```CLIP``` will be merged into the base script.
- ```utils.py``` contains useful functions. Notably, this is where augmentations and dataloader creation is defined. If new augmentations are to be used, add them here. Currently, only cifar10/stl10 are supported as ID/OOD. New datasets can be added here. 
- ```extract_cka.py``` computes the CKA scores of a given model(s) for specified dataset(s). Use the command line args to specify what should be compared: modelA vs. modelB, datasetA vs. datasetB. Specify what layers to compared in `layer_names`.  
- ```extract_prediction_depth.py``` creates the kNN index probes for computing prediction depth. This file is used to create and save the probes. Use ```eval_prediction_depth.py``` to compute the depth. Please note, that computing the depth is memory intensive -- specify the layers to extract and create probe in ```layer_names.``` 
- ```eval_prediction_depth.py```  given the trained kNN index probes, model and dataset, computes the prediction depth. 

Results after training are saved to ```logs/```. There is a consolidated file across all experiments and a log file for each experiment (epoch, train loss, test loss, test error).

Paths for saving and loading data will need to be updated. I've tried to include a ```PREFIX``` variable to do this in key files. However, you may manually need to adjust the paths for whatever dataset is being used. 


## Hyper-parameter Tuning for Domainnet/CLIP
Using the files found in ```clip_scripts``` it should be easy to launch a search over the learning rate for the ```FT``` and ```LP+FT``` protocols. ```ft_train_aug``` should be set to base when performing ```FT```. A ```LP``` checkpoint trained with ```base``` should be used as the starting checkpoint for the ```LP+FT``` protocol (set ```ft_train_aug=base``` ).

I suspect that the learning rate needs to be very small for ```lp+ft``` and ```ft``` to do well. ```[3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]```.

### Acknowledgements 
This code base is inspired by and makes us of several great code bases. We especially thank the authors of [PixMix](https://github.com/andyzoujm/pixmix) and [Fine Tuning can Distort...](https://github.com/AnanyaKumar/transfer_learning). 