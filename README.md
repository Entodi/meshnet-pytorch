# MeshNet 

This repository contains PyTorch implementation of MeshNet architecture. MeshNet is volumetric convolutional neural network based on dilated kernels [1] for image segmentation. 

Code provides framework to train, evaluate model for segmentation of 104 class brain atlas. It is modification of our previous work [3]. 

# Usage
## Data preparation
1. Prepare **T1** input with mri_convert from FreeSurfer (https://surfer.nmr.mgh.harvard.edu/) conform T1 to 1mm voxel size in coronal slice direction with side length 256. **You can skip this step if your T1 image with slice thickness 1mm x 1mm x 1mm and 256 x 256 x 256.**
```
mri_convert *brainDir*/t1.nii *brainDir*/T1.nii.gz -c
```
2. Prepare **labels** from aparc+aseg.nii.gz using:
```
python prepare_data.py --brains_list *brains_lits.txt*
```

## Training

To train the model use following command:
```
python main.py --model ./models_configurations/MeshNet_104_38_T1.yml --train_path ./folds/hcp_example/train.txt --validation_path ./folds/hcp_example/validation.txt
```

We also support Visdom (https://github.com/facebookresearch/visdom) monitoring for training. To use it use arguments: 
```
--visdom --visdom_server *[visdom_server_ip]* --visdom_port *[visdom_server_port]*
```

## Evaluation
To evaluate the model use the following command:
```
python evaluation.py --models_file example_models.txt --evaluation_path folds/hcp_example/test.txt
```

# Requirements

* Install PyTorch https://pytorch.org/get-started/locally/
* Install other dependicies
```
pip install -r requirements.txt
```

# Acknowledgment

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

# References

[1] https://arxiv.org/abs/1511.07122 Multi-Scale Context Aggregation by Dilated Convolutions. *Fisher Yu, Vladlen Koltun*  
[2] https://arxiv.org/abs/1612.00940 End-to-end learning of brain tissue segmentation from imperfect labeling. *Alex Fedorov, Jeremy Johnson, Eswar Damaraju, Alexei Ozerin, Vince D. Calhoun, Sergey M. Plis*  
[3] https://arxiv.org/abs/1711.00457 Almost instant brain atlas segmentation for large-scale studies. Alex Fedorov, Eswar Damaraju, Vince Calhoun, Sergey Plis  
[4] http://www.humanconnectomeproject.org/ Human Connectome Project  

# Previously

## Brain Atlas segmentation with [**brainchop.org**](http://brainchop.org)
To get brain atlas segmentation ([3]) you don't need to run any code. Just sign up at [**brainchop.org**](http://brainchop.org), upload your 3T MRI T1 image and get brain atlas in 1-2 minutes.

Watch video with example of brain atlas segmentation.  
[![IMAGE ALT TEXT](http://img.youtube.com/vi/Nc-l1qd3dAg/0.jpg)](https://www.youtube.com/embed/Nc-l1qd3dAg?autoplay=1&loop=1&playlist=Nc-l1qd3dAg)

## Torch Implementation for brain tissue segmentation https://github.com/Entodi/MeshNet

### Result on subject **105216**
| T1 MRI  | FreeSurfer | MeshNet |
|---|---|---|
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_t1.gif?raw=true)  |  ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_fs.gif?raw=true)   | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_219.gif?raw=true)  |

# Questions

You can ask any questions about implementation and training by sending message to **eidos92@gmail.com**.
