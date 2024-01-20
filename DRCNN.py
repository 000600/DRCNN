import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Define paths
train_path = ' < PATH TO TRAIN DATA > '
test_path = ' < PATH TO TEST DATA > '
val_path = ' < PATH TO VALIDATION DATA > '

# Set batch size and epochs
batch_size = 64
epochs = 10

# Load training data
train_generator = ImageDataGenerator(rescale = 1 / 255, zoom_range = 0.01, rotation_range = 0.05, width_shift_range = 0.05, height_shift_range = 0.05)
train_iter = train_generator.flow_from_directory(train_path, class_mode = 'binary', color_mode = 'rgb', batch_size = batch_size)

# Load test data
test_generator = ImageDataGenerator(rescale = 1 / 255)
test_iter = test_generator.flow_from_directory(test_path, class_mode = 'binary', color_mode = 'rgb', batch_size = batch_size)

# Load validation data
val_generator = ImageDataGenerator(rescale = 1 / 255)
val_iter = val_generator.flow_from_directory(val_path, class_mode = 'binary', color_mode = 'rgb', batch_size = batch_size)

# Define classes
class_map = train_iter.class_indices

# Initialize Adam optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Input layer
model.add(Input(train_iter.image_shape))

# Image processing layers
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))
model.add(Conv2D(filters = 30, kernel_size = (3, 3), activation = 'relu'))

# Flatten layer
model.add(Flatten())

# Hidden layers
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

# Output layer
model.add(Dense(1, activation = 'sigmoid'))  # Sigmoid activation function since the model is binary

# Configure early stopping
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Compile and train model
model.compile(optimizer = opt, loss = BinaryCrossentropy(), metrics = [BinaryAccuracy(), AUC()])
history = model.fit(train_iter, steps_per_epoch = int(round(train_iter.samples / train_iter.batch_size)), epochs = epochs, validation_data = val_iter, validation_steps = int(round(val_iter.samples / batch_size)))

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['binary_accuracy']
val_accuracy = history_dict['val_binary_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize AUC and validation AUC
auc = history_dict['auc']
val_auc = history_dict['val_auc']

plt.plot(auc, label = 'Training AUC')
plt.plot(val_auc, label = 'Validation AUC')
plt.title('Validation and Training AUC Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# View train, test, and validation accuracy
train_loss, train_acc, train_auc = model.evaluate(train_iter, verbose = 0) # Change verbose to 1 or 2 for more information
test_loss, test_acc, test_auc = model.evaluate(test_iter, verbose = 0) 
val_loss, val_acc, val_auc = model.evaluate(val_iter, verbose = 0)

print(f'Train accuracy: {train_acc * 100}%')
print(f'Test accuracy: {test_acc * 100}%')
print(f'Validation accuray: {val_acc * 100}%')

# Get inputs to view the model's predictions compared to actual labels
sample_inputs, sample_labels = val_iter.next()

# Change this number to view more or less input images and corresponding predictions and labels
num_viewed_inputs = 10

# Get inputs and corresponding labels and predictions
sample_inputs = sample_inputs[:num_viewed_inputs]
sample_labels = sample_labels[:num_viewed_inputs]
sample_predictions = model.predict(sample_inputs)

# Combine lists
img_pred_label = enumerate(zip(sample_inputs, sample_predictions, sample_labels))

# Create a reverse class map that can be used to access the class based on its numerical encoding
reverse_class_map = {c : k for k, c in class_map.items()}

# Loop through combined list to display the image, the model's prediction on that image, and the actual label of that image
for i, (img, pred, label) in img_pred_label:
  # Model's prediction on sample photo
  predicted_class = 1 if pred >= 0.5 else 0 # Round the model's prediction; a prediction of 0.5 or greater signifies a prediction of class 1 (no DR), a prediction of less than 0.5 signifies a prediction of class 0 (DR detected)
  certainty = 1 - pred[0] if predicted_class == 0 else pred[0]
  # Actual values
  actual_class = label

  # View results
  print(f"Model's Prediction ({certainty * 100}% certainty): {predicted_class} ({reverse_class_map[predicted_class]}) | Actual Class: {actual_class} ({reverse_class_map[actual_class]})\n")

  # Visualize input images
  plt.axis('off')
  plt.imshow(img[:, :, 0])
  plt.tight_layout()
  plt.show()
