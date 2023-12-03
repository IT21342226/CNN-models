
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

dataset_dir = 'D:\\Datasets\\Chessman-image-dataset'



# Define the ImageDataGenerator for preprocessing and data augmentation
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

# Create a flow from the directory for both training and testing
train_generator = datagen.flow_from_directory(
    "D:\\Datasets\\Chessman-image-dataset\\Chess",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # use 'categorical' for multiple classes
    subset='training',
    seed=100,
    shuffle=True 
)

test_generator = datagen.flow_from_directory(
    "D:\\Datasets\\Chessman-image-dataset\\Chess",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load the VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding your own fully connected layers on top of the VGG16 base
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
history = model.fit(train_generator, epochs=15, validation_data=test_generator)

model.evaluate(test_generator)

import matplotlib.pyplot as plt

def plot_acc_loss(history, epochs):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_acc_loss(history, 15)

model.save('chess_find.keras')

from tensorflow.keras.models import load_model

loaded_model = load_model('chess_find.keras')

get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'loaded_model' is your pre-trained model
# Modify the path to your saved model file if needed
loaded_model = tf.keras.models.load_model('path/to/your/model.h5')

img_path = 'D:/Datasets/Chessman-image-dataset/Chess/test/king/00000023.png'  # Change this to the path of your test image

img1 = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img1)
img = img / 255
img = np.expand_dims(img, axis=0)

prediction = loaded_model.predict(img, batch_size=None, steps=1)

class_labels = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']  # Modify based on your class order

predicted_class = class_labels[np.argmax(prediction)]

plt.imshow(img1)
plt.title(f'Predicted Class: {predicted_class}')
plt.show()

