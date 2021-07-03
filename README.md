# deep-learning-for-pkd-patients

## Goal
The aim of this research is to predict the disease progression in PKD patients. We will be using deep learning to carry out this task.

## Preprocessing
Our dataset consists of DICOM files. We need to preprocess these because the images might have different dimensions. Also we might have to rescale them and try to preserve maximum data. We might therefore have to resample our images. 

The pydicom library is used to work with these files. We will convert that to numpy arrays that will be passed to our NN.

https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet

https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

## Running instructions
* This script runs on Python 3. It can be installed from https://www.python.org/downloads/
* Clone this repository using: git clone https://github.com/amalkoodoruth/deep-learning-for-pkd-patients.git
* Open terminal (or command prompt on windows or similar) and change your directory to: .../deep-learning-for-pkd-patients
* install the required libraries by running '''sh
pip '''
* Run the python script using: python main.py

## Expected output
The script should create a file named "mri_scan.png" and it should look like ![](https://github.com/amalkoodoruth/deep-learning-for-pkd-patients/blob/main/expected_output.png)

## Procedure
7/1/21: As a first step, we will train a 2D segmentation model on the CHAOS dataset. We are just trying to get some code running while waiting for our real dataset.


## 3D Neural Network
How does it work? 

One method is to pass each slide in from 1 image in the neural network. Then predict for each slice, aggregate results and take argmax. So for example if a 3D scan has dimensions \[100, (224, 224)\], we will pass 100 224 x 224 images in the network. Our prediction, for example if we have to do 2-way classification, will be a one-hot encoded array: \[1,0\] if yes and \[0,1\] if no. Then we divide by 100 then take the argmax. 

How will training work? 

Taking the two-way classification example, when we train the network, \[.01,.99\] should be penalized more than \[0.49,0.51\] if the prediction is wrong. Note that both predict class 2. 

Theory behind that? Pytorch or tensorflow documentation.......

## Data augmentation
We might want to add noise to increase the size of our dataset. But we should be careful in not adding too much. 



