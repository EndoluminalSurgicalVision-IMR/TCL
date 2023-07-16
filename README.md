# TCL
TCL: Triplet Consistent Learning for Odometry Estimation of Monocular Endoscope
<img src="assets/TCL_framework.png"  alt="" align=center />



## Introduction 
This is the official implementation of MICCAI 2023 paper "TCL: Triplet Consistent Learning for Odometry Estimation of Monocular Endoscope".
    
    Triplet-Consistent-Learning framework (TCL) consisting of two
    modules: Geometric Consistency module(GC) and Appearance Inconsis-
    tency module(AiC). To enrich the diversity of endoscopic datasets, the
    GC module generates synthesis triplets and enforces geometric consis-
    tency via specific losses. To reduce the appearance inconsistency in the
    image triplets, the AiC module introduces a triplet-masking strategy to
    act on photometric loss. TCL can be easily embedded into various unsu-
    pervised methods without adding extra model parameters.

This project provides the GC and AiC modules, which can be applied to SfM-based methods to improve their performance on endoscopic datasets.

## Preparation

### Environment Setup
    python=3.6
    scipy=1.5.4
    numpy=1.19.5
    torch=1.10.0
    
    
### Pseudo-depth label preparation
Train a state-of-the-art SfM-based method(Monodepth2,SC-SfMlearner,etc..) on the training dataset. Save the trained models and use them to generate pseudo-depth labels for the images in the training set. 

### Dataset & SfM-based Unsupervised Method
Prepare an endoscopic dataset and an SfM-based unsupervised method.


## Usage
Our two modules can be embedded into the SfM-based unsupervised method to increase the diversity of endoscopic samples and to reduce the impact of appearence inconsistency in endoscopic triplets on training.

You can adapt the two modules to your SfM-based baseline by following the hints and comments in the code.

### Modules
The TCL and AiC modules are provided in TCL.py and AiC.py, respectively.

### Tips on training
To achieve better results, we would like to share some training tips when embedding TCL into the baseline:
#### Tips on TCL
1. It is important to assess whether the current endoscopic dataset is suitable for data augmentation. If the dataset size is already sufficient, data augmentation may not necessarily lead to improved results.
2. When incorporating TCL, we recommend starting with training without including the two consistency losses. Initially, focus on performing perturbed data augmentation to determine appropriate perturbation bound for data augmentation.
3. Once the perturbation bound have been determined, introduce the depth consistency loss and experiment with different weights on a larger scale. This step aims to improve depth estimation while avoiding overfitting.
4. After establishing the depth consistency loss, incorporate the pose consistency loss while keeping the depth consistency loss intact. Experiment with a wide range of weights for the pose consistency loss to find the optimal balance.
5. It is important to note that the optimal weights for the consistency losses may vary significantly depending on the specific dataset and baseline model being used. Therefore, it is crucial to carefully adjust the weights according to the characteristics of the dataset and baseline model to achieve the best performance.

#### Tips on AiC
1. The AiC module is designed to reduce the impact of appearance inconsistency on the photometric loss in triplets. Therefore, it is important to assess whether there is significant appearance inconsistency in your training dataset. Based on the severity of the inconsistency, you can experiment with different lower bounds for the triplet mask (tmthre).
2. To evaluate the presence of appearance inconsistency, you can observe variations in reflections and brightness or analyze the distribution of the photometric loss across the entire image. These indicators can help measure the level of appearance inconsistency and guide the selection of an appropriate triplet mask lower bound (tmthre).

### Related projects
Thank the authors for their superior works: 
[MonoDepth2](https://github.com/nianticlabs/monodepth2), [AF-SfMleaner](https://github.com/shuweishao/af-sfmlearner),[CPP](https://github.com/yzhao520/CPP),[SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release),[EndoSLAM](https://github.com/CapsuleEndoscope/EndoSLAM)



