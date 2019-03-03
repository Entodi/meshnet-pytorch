# MeshNet 

This repository contains PyTorch implementation of MeshNet architecture. MeshNet is volumetric convolutional neural network based on dilated kernels [1] for image segmentation. 

Code provides framework to train, evaluate model for segmentation of 104 class brain atlas. It is modification of our previous work [4]. We also using Cosine Annealing scheduler with Warm Restarts [5] (**Experimental**).

# Training
```
python main.py --model ./models_configurations/MeshNet_104_38_T1.yml --train_path ./folds/hcp_example/train.txt --validation_path ./folds/hcp_example/validation.txt --batch_size 8 --sv_w 38 --sv_h 38 --sv_d 38 --n_threads [n_threads] --weight_init xavier_normal --visdom_server [visdom_server url]
```

# Evaluation
```
python evaluation.py --models_file example_models.txt --evaluation_path folds/hcp_example/test.txt --batch_size 8 --n_subvolumes 1024
```

# Requirements

* Install PyTorch https://pytorch.org/get-started/locally/
* Install other dependicies
```
pip install -r requirements.txt
```

# Acknowledgment

# References

[1] https://arxiv.org/abs/1511.07122 Multi-Scale Context Aggregation by Dilated Convolutions. *Fisher Yu, Vladlen Koltun*  
[2] https://arxiv.org/abs/1612.00940 End-to-end learning of brain tissue segmentation from imperfect labeling. *Alex Fedorov, Jeremy Johnson, Eswar Damaraju, Alexei Ozerin, Vince D. Calhoun, Sergey M. Plis*  
[3] http://www.humanconnectomeproject.org/ Human Connectome Project  
[4] https://arxiv.org/abs/1711.00457 Almost instant brain atlas segmentation for large-scale studies. Alex Fedorov, Eswar Damaraju, Vince Calhoun, Sergey Plis  
[5] https://arxiv.org/abs/1608.03983 SGDR: Stochastic Gradient Descent with Warm Restarts. Ilya Loshchilov, Frank Hutter

# Previously

## Brain Atlas segmentation with [**brainchop.org**](http://brainchop.org)
To get brain atlas segmentation (https://arxiv.org/abs/1711.00457 extension of this work) you don't need to run any code. Just sign up at [**brainchop.org**](http://brainchop.org), upload your 3T MRI T1 image and get brain atlas in 1-2 minutes.

Watch video with example of brain atlas segmentation.  
[![IMAGE ALT TEXT](http://img.youtube.com/vi/Nc-l1qd3dAg/0.jpg)](https://www.youtube.com/embed/Nc-l1qd3dAg?autoplay=1&loop=1&playlist=Nc-l1qd3dAg)

## Torch Implementation for brain tissue segmentation https://github.com/Entodi/MeshNet

Result on subject **105216**  

| T1 MRI  | FreeSurfer | MeshNet |
|---|---|---|
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_t1.gif?raw=true)  |  ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_fs.gif?raw=true)   | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_219.gif?raw=true)  |

# Questions

You can ask any questions about implementation and training by sending message to **eidos92@gmail.com**.
