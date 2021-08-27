# deep-learning-for-pkd-patients

## Goal
The aim of this research is to predict the disease progression in PKD patients. Because of time constraints, we have only built a beta model that works on public data available at: 
https://chaos.grand-challenge.org/

We will be using deep learning to carry out the segmentation of 2D images. The accuracy of the model is determined using the dice coefficient. 

## Preprocessing
Our dataset consists of DICOM files. We need to preprocess these because the images might have different dimensions. Also we might have to rescale them and try to preserve maximum data. We might therefore have to resample our images. Right now, we are just padding our images with 0s. The images that are fed to the model have dimensions 320x320. Note that it is important that the dimensions are divisible by 16, because of the UNET architecture that we are using. 

The pydicom library is used to work with these files. We will convert that to numpy arrays that will be passed to our neural network.

https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet

https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

Data augmentation techniques including flips are implemented to obtain a more general model.


## Running instructions for testing Dataset class and UNET
* This script runs on Python 3. It can be installed from https://www.python.org/downloads/
* Open terminal (or command prompt on windows or similar) and clone this repository using: ```git clone https://github.com/amalkoodoruth/deep-learning-for-pkd-patients.git```
* Change your directory to: .../deep-learning-for-pkd-patients/beta
* install the required libraries by running ```$ pip install -r requirements.txt ```. This command will install the following libraries:
	- matplotlib~=3.4.2
	- numpy==1.19.2
	- pandas~=1.2.5
	- Pillow~=8.3.0
	- pydicom~=2.1.2
	- torch~=1.9.0
	- torchvision~=0.10.0
	- tqdm~=4.61.2
* Change directory to .../deep-learning-for-pkd-patients/beta
* Run the python script using: python train.py


An example notebook is provided in the "beta" folder.


## TO BE IMPLEMENTED
### 3D Neural Network
How does it work? 

One method is to pass each slide in from 1 image in the neural network. Then predict for each slice, aggregate results and take argmax. So for example if a 3D scan has dimensions \[100, (224, 224)\], we will pass 100 224 x 224 images in the network. Our prediction, for example if we have to do 2-way classification, will be a one-hot encoded array: \[1,0\] if yes and \[0,1\] if no. Then we divide by 100 then take the argmax. 

### Nested cross validation
This technique allows us to train our model on the whole dataset. The inner loop takes care of the hyperparameter tuning while the outer loop gives us the possibility to train the model on the whole dataset. Then, we select the best model.

### Predictor
Function that takes in the path to MRI scans of a patient and outputs the segmented parts.

### Preporocessing data
Outlining of cyst and kidney.

## References

1. https://github.com/aladdinpersson/Machine-Learning-Collection
2. https://arxiv.org/abs/1505.04597
3. https://www.nature.com/articles/s41598-020-77981-4.pdf
4. https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

