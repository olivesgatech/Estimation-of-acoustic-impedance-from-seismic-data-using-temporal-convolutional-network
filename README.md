# Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network
Ahmad Mustafa, [Motaz Alfarraj](http://www.motaz.me), and [Ghasssan AlRegib](http://www.ghassanalregib.com) 

This repository includes the codes for the paper:

'**Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network**', Expanded Abstracts of
 the SEG Annual Meeting , San Antonio, TX, Sep. 15-20, 2019. [[PDF]](https://arxiv.org/abs/1906.02684)

The code has been built in Python 3.7.0 and requires the following dependencies before it can be run:
```
numpy 1.15.1
matplotlib 3.0.0
pytorch 0.4.1
seaborn 0.9.0
segyio 1.7.0
tensorboard 1.13.1
tensorboardx 1.7
tensorflow 1.13.1
pandas 0.24.2
scipy 1.1.0
```

## Abstract

In exploration seismology, seismic inversion refers to the process of inferring physical properties of the subsurface 
from seismic data. Knowledge of physical properties can prove helpful in identifying key structures in the subsurface 
for hydrocarbon exploration. In this work, we propose a workflow for predicting acoustic impedance (AI) from seismic 
data using a network architecture based on Temporal Convolutional Network by posing the problem as that of sequence 
modeling. The proposed workflow overcomes some of the problems that other network architectures usually face, like 
gradient vanishing in Recurrent Neural Networks, or overfitting in Convolutional Neural Networks. The proposed workflow
was used to predict AI on Marmousi 2 dataset with an average r<sup>2</sup> coefficient of 91% on a hold-out validation set. 
 
## Dataset
Create a directory called `data` inside the parent directory containing the project files. Download the data from this 
[link](https://www.dropbox.com/sh/caxmz94vo3ms2bl/AABTWb2KRrKzLRinfwG0G7UGa?dl=0) and put the contents in the `data` folder.

## Running the Code
After scrolling to the directory containing the project, in the command line, run:
```
python train.py 
```
This will run the *train.py* file with the default settings for all the parameters. To view all these along with their 
default values, run:
```
python train.py -h
```
If you want to change, say the number of epochs to 1200, just run:
```
python train.py --n_epochs 1200
```

### Inspecting the Results
The training and validation losses will be printed on the terminal along with the epoch number. After the training is 
over, the images for the ground-truth AI model, the predicted AI model, the absolute difference between the two, and a 
figure showing hand-picked predicted AI traces superimposed on their ground-truth traces will be printed on the screen.
You can also view the training and validation loss curves along with the predicted AI images in *tensorboard* by 
running:
```
tensorboard --logdir path/to/log-directory
```  

## Citation 
If you have found our code and data useful, we humbly request you to cite our work. You can cite the arXiv preprint:
```tex
@incollection{amustafa2019AI,
title=Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network,
author=Mustafa, Ahmad and AlRegib, Ghassan,
booktitle=arXiv:1906.02684,
year=2019,
publisher=Society of Exploration Geophysicists}
```
The arXiv preprint is available at: [https://arxiv.org/abs/1906.02684](https://arxiv.org/abs/1906.02684)

## Questions?
The code and the data are provided as is with no guarantees. If you have any questions, regarding the dataset or the 
code, you can contact me at (amustafa9@gatech.edu), or even better, open an issue in this repo and we will do our best 
to help. 
