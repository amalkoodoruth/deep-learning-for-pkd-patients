# deep-learning-for-pkd-patients

## Goal
The aim of this research is to predict the disease progression in PKD patients. We will be using deep learning to carry out this task.

## Preprocessing
Our dataset consists of DICOM files. We need to preprocess these because the images might have different dimensions. Also we might have to rescale them and try to preserve maximum data. We might therefore have to resample our images. 

The pydicom library is used to work with these files. We will convert that to numpy arrays that will be passed to our NN.

https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet

https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial

## Procedure
Once all samples in our data have the same dimensions, they will be ready to be fed to the network.

While we are not very sure of what exactly we want to do right now (June 9th), we believe that we need to first isolate the kidney. 

So the first step would be segmentation. We will need to have in-house experts to segment the kidneys. That would give us our training, validation and testing set.
Next, once we obtain the kidney images only, these will be fed to a NN that will classify the patient in classes 1A-1E. 

Does predicting volume give us a better result? Don't know yet but this might be useful.

Then, to get disease progression, we would need eGFR. Will our model do this? How are we going to do that? Needs to be discussed with Dr Alam on Friday June 11th.
