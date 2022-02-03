# MultiWienerNet
## Deep learning for fast spatially-varying deconvolution 

### [Project Page](https://waller-lab.github.io/MultiWienerNet/) | [Paper](https://doi.org/10.1364/OPTICA.442438)

### Setup:
Clone this project using:
```
git clone https://github.com/Waller-Lab/MultiWienerNet.git
```

Install dependencies. We provide code both in Tensorflow and in Pytorch. Tensorflow version only contains an implementation for 2D deconvolutions, whereas the Pytorch contains both 2D and 3D deconvolutions. 

If using Pytorch, install the depencies as:

```
conda env create -f environment.yml
source activate multiwiener_pytorch
```

If using Tensorflow, install the depencies as:

```
conda env create -f environment.yml
source activate multiwiener_tf
```

## Using pre-trained models
We provide an example of a pre-trained MutliWienerNet for fast 2D deconvolutions as well as compressive 3D deconvolutions from 2D measurements. These examples are based on data for Randoscope3D. To adapt this model to your own data, please see [below](#training_link). 


### Loading in pretrained models
The pre-trained models can be downloaded: [here (pytorch)](https://drive.google.com/drive/folders/1teIPp2q2ce0l9FjYe0LuC9c-Rpq2fA8x?usp=sharing) and [here (tensorflow)](https://drive.google.com/drive/folders/1E3bye75ovDvfKsDG4IMe_hzo5wQU1zTP?usp=sharing) 
Please download these and place them in the pytorch/saved_models and tensorflow/saved_models

### Loading in data 
We provide several limited examples of 2D and 3D data in /data/
You can download the full dataset that we have used for training [here](link). 