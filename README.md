## FG-TTT Project
A Test-Time Training project built on nnUNet v1.7.1, designed for medical image segmentation with multi-view consistency constraints.


## Core Scripts
The key implementation files include:
- `FG_TTT_nnunet/training/network_training/umd_trainer.py`  
- `FG_TTT_nnunet/training/network_training/refiner.py`  
- `FG_TTT_nnunet/training/dataloading/umd_dataloader.py`  
- `FG_TTT_nnunet/training/loss_functions/multiview_consistency_loss.py`  

## Environment configuration
### Step 1: Clone the repository:

git clone https://github.com/JunhaoWang-bme/FG-TTT.git
cd FG-TTT


### Step 2: Install dependencies:

pip install -r requirements.txt


## Data Preparation
The data preprocessing pipeline follows the same approach as the standard nnUNet framework.
Key Note on Data Naming Format
All data files must adhere to the following naming convention:UMD_(id)_(plane).nii.gz
(id): Unique identifier for the subject/sample.
(plane): Imaging plane (e.g., sagittal, transverse, coronal) corresponding to the data.
This format ensures compatibility with the custom dataloader (umd_dataloader.py) and downstream processing steps.

```
nnUNet_raw_data/TaskXXX_UMD/
├── imagesTr/
│   ├── UMD_001_sag.nii.gz
│   ├── UMD_001_cor.nii.gz  # Not Necessary
│   ├── UMD_001_tra.nii.gz  # Not Necessary
│   └── ...
├── labelsTr/               # placeholder folder (The labels file is just a placeholder and does not participate in supervised training, pseudo-labels can be generated with pre-trained models.)
└── dataset.json            # standard nnUNet format
```

## Running the Project

### Prerequisite  
Before executing the project, you need to load pre-trained weights (train a segmentation model using nnUNet v1.7.1 first).  


### Step 1: Convert Task Dataset  
Convert the dataset to nnUNet-compatible format:  

nnUNet_convert_decathlon_task -i INPUT_FOLDER/TaskXXX_UMD


### Step 2: Plan and Preprocess  
Generate training plans and preprocess the dataset:  

nnUNet_plan_and_preprocess -t XX --verify_dataset_integrity


### Step 3: Train with Pre-trained Weights  
Train the model using the custom trainer and pre-trained weights:  

nnUNet_train 3d_fullres UMDConsistencyTrainer XXX 4 --npz -pretrained_weights /path/to/model_best_checkpoint.model


### Step 4: Inference/Prediction  
Run inference on new data:  

nnUNet_predict -i INPUT -o OUTPUT -t XXX -m 3d_fullres -tr UMDConsistencyTrainer -f 4


## About nnInteractive
The items and weights can be downloaded from the nnInteractive model website: https://github.com/MIC-DKFZ/nnInteractive