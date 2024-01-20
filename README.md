# Diabetic Retinopathy CNN

## The Neural Network
This convolutional neural network predicts whether a patient is healthy or if they have diabetic retinopathy based on images of their retinas. The model will predict a value close to 0 if the patient is predicted to have diabetic retinopathy and a 1 if the patient is predicted not to have diabetic retinopathy. Since the model only predicts binary categorical values, it uses a binary cross-entropy loss function and has 1 output neuron with a sigmoid activation function. The model uses a standard Adam optimizer with a learning rate of 0.001 and implements a dropout layer and early stopping to prevent overfitting. The model uses Tensorflow's **ImageDataGenerator** to rescale the data and has an architecture consisting of:
- 1 Input layer (with an input shape of (256, 256, 3))
    * The input shape has three color channels because the retinal images are in-color
- 1 Conv2D layer (with 32 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Max pooling 2D layer (with a pooling size of (2, 2) and strides of (2, 2) and "valid" padding)
- 1 Conv2D layer (with 30 filters, a kernel size of (3, 3), and a ReLU activation function)
- 1 Flatten layer
- 3 Hidden layers (each with either 256, 128, or 64 neurons and a ReLU activation function)
- 1 Dropout layer (in between the first two hidden layers and with a dropout rate of 0.4)
- 1 Output layer (with 1 neuron and a sigmoid activation function)

I found the neural network had consistent test and validation accuracies of around 94% and an AUC of around 98% after I tuned various hyperparameters, but I am sure higher accuracies are possible. Feel free to further tune the hyperparameters or build upon the model! 

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy/data. Credit for the dataset collection goes to **pkdarabi**, **markkostantine**, **nickchotik**, and others on *Kaggle*. In addition to containing more information about diabetic retinopathy, the dataset contains approximately 2076 training images, 231 test images, and 531 validation images (2838 images total). The dataset is not included in the repository because it is too large to stably upload to Github, so use the link above to find and download the dataset.

To properly load the dataset into the model, you will need to input the paths to the training, testing, and validation datasets (three paths total) as strings into the **DRCNN.py** file. These paths need to be inputted where the file reads:
- " < PATH TO TRAIN DATA > " 
- " < PATH TO TEST DATA > "
- " < PATH TO VALIDATION DATA > " 

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way.
