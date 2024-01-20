# Diabetic Retinopathy CNN

## The Model

This convolutional neural network predicts whether a patient is healthy or if they have diabetic retinopathy based on a scan of their retinas. The model will predict a value close to 0 if the patient is predicted to have diabeteic reintopahty and a 1 if the patient is predicted not to have diabetic retinopathy. Since the model only predicts binary categorical values, it uses a binary crossentropy loss function and has 1 output neuron with a sigmoid activation function. The model uses a standard Adam optimizer with a learning rate of 0.001 and a dropout layer to prevent overfitting. The model uses Tensorflow's **ImageDataGenerator** to rescale the data and has an architecture consisting of:
- 1 Input layer (with an input shape of (256, 256, 1))
    * The images only have one color channel because they are considered grayscale
- 1 Conv2D layer (with 32 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Flatten layer
- 3 Hidden layers (each with either 128, 64, or 32 neurons and a ReLU activation function)
- 1 Dropout layer (in between the two hidden layers and with a dropout rate of 0.2)
- 1 Output neuron (with 4 neurons and a softmax activation function)

When running the **chest_xray_cnn.py** file, you will need to input the paths of the training, testing, and validation datasets (three paths total) as strings. These paths need to be inputted where the file reads:
- " < PATH TO TRAIN SET IMAGES > " 
- " < PATH TO TEST SET IMAGES > "
- " < PATH TO VALIDATION SET IMAGES > " 

I found the neural network had consistent test and validation accuracies of around 94% and an AUC of around 98% after I tuned various hyperparameters, but I am sure higher accuracies are possible. Feel free to further tune the hyperparameters or build upon the model!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis. Credit for the dataset collection goes to **Hubert Serowski**, **Khizar Khan**, **Adryn H.**, and others on *Kaggle*. The dataset contains approximately 6566 training images, 801 testing images. and 48 validation images (7135 images total). Note that the images from the original dataset are considered grayscale. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## The Dataset
https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/code

## Libraries
These neural networks and XGBoost Regressor were created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way.
