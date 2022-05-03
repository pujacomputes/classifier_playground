## Instructions

```
pip install torch==1.11.0
mkdir logs
mkdir ckpts
cd scripts
./run_ft.sh
```

- ```train.py``` is the main file, which accepts command line arguments to run different protocols (lp, ft, lp+ft) using different augmentation strategies. Note: to replicate the feature distortion paper, the ```train_aug``` and ```ft_train_aug``` should be set to ```test```. Doing so, prevents horizontal flipping and cropping from being used as augmentations.  
- ```utils.py``` contains useful functions. Notably, this is where augmentations and dataloader creation is defined. If new augmentations are to be used, add them here. Currently, only cifar10/stl10 are supported as ID/OOD. New datasets can be added here. 
- ```extract_cka.py``` computes the CKA scores of a given model(s) for specified dataset(s). Use the command line args to specify what should be compared: modelA vs. modelB, datasetA vs. datasetB. Specify what layers to compared in `layer_names`.  
- ```extract_prediction_depth.py``` creates the kNN index probes for computing prediction depth. This file is used to create and save the probes. Use ```eval_prediction_depth.py``` to compute the depth. Please note, that computing the depth is memory intensive -- specify the layers to extract and create probe in ```layer_names.``` 
- ```eval_prediction_depth.py```  given the trained kNN index probes, model and dataset, computes the prediction depth. 

Results after training are saved to ```logs/```. There is a consolidated file across all experiments and a log file for each experiment (epoch, train loss, test loss, test error).
