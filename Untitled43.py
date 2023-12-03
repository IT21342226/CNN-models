
import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set the path to your dataset directory
dataset_dir = 'D:\\root\\data'

w_maskDir  = os.path.join(dataset_dir, 'with_mask')
n_maskDir = os.path.join(dataset_dir, 'without_mask')

# Get the list of all image files in each category
w_mask_file = [os.path.join(w_maskDir, file) for file in os.listdir(w_maskDir) if file.endswith('.jpg')]
n_mask_file = [os.path.join(n_maskDir, file) for file in os.listdir(n_maskDir) if file.endswith('.jpg')]


# Shuffle the file lists to ensure randomness
random.shuffle(w_mask_file)
random.shuffle(n_mask_file)

# Define the ratio of data to be used for training
train_ratio = 0.8

# Split the data into training and testing sets
w_mask_train_files = w_mask_file[:int(train_ratio * len(w_mask_file))]
w_mask_test_files = w_mask_file[int(train_ratio * len(w_mask_file)):]

n_mask_train_files = n_mask_file[:int(train_ratio * len(n_mask_file))]
n_mask_test_files = n_mask_file[int(train_ratio * len(n_mask_file)):]

# Define the ImageDataGenerator for preprocessing and data augmentationb
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Create a directory for both training and testing data in the specified path
train_test_dir = 'D:\\root\\train_test'
os.makedirs(train_test_dir, exist_ok=True)

# Create directories for each class in training and testing sets
train_dir = os.path.join(train_test_dir, 'train_data')
test_dir = os.path.join(train_test_dir, 'test_data')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Create directories for each class inside the training directory
w_mask_train_dir = os.path.join(train_dir, 'with_mask')
n_mask_train_dir = os.path.join(train_dir, 'without_mask')

os.makedirs(w_mask_train_dir, exist_ok=True)
os.makedirs(n_mask_train_dir, exist_ok=True)

# Create directories for each class inside the testing directory
w_mask_test_dir = os.path.join(test_dir, 'with_mask')
n_mask_test_dir = os.path.join(test_dir, 'without_mask')

os.makedirs(w_mask_test_dir, exist_ok=True)
os.makedirs(n_mask_test_dir, exist_ok=True)

# Copy training files to class-specific directories in the training directory
for i, file_path in enumerate(w_mask_train_files):
    shutil.copy(file_path, os.path.join(w_mask_train_dir, f"{i}.jpg"))

for i, file_path in enumerate(n_mask_train_files):
    shutil.copy(file_path, os.path.join(n_mask_train_dir, f"{i}.jpg"))

# Copy testing files to class-specific directories in the testing directory
for i, file_path in enumerate(w_mask_test_files):
    shutil.copy(file_path, os.path.join(w_mask_test_dir, f"{i}.jpg"))

for i, file_path in enumerate(n_mask_test_files):
    shutil.copy(file_path, os.path.join(n_mask_test_dir, f"{i}.jpg"))


img_width = 300
img_height = 300
batch_size = 64


train_generator = datagen.flow_from_directory(
    'D:\\root\\train_test\\train_data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)


test_generator = datagen.flow_from_directory(
    'D:\\root\\train_test\\test_data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Count the number of images in training and testing sets
num_train_images = len(train_files)
num_test_images = len(test_files)

print(f"Number of training images: {num_train_images}")
print(f"Number of testing images: {num_test_images}")

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


classifier = Sequential()
classifier.add(Conv2D(32,(4,4),input_shape=(300,300,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
classifier.add(Conv2D(32,(4,4),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
adam = tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


# Specify the number of training steps per epoch
steps_per_epoch = num_train_images // batch_size

# Fit the model using the data generator
history = classifier.fit_generator(
    train_generator,
    steps_per_epoch=300,
    epochs=20,
    validation_data=test_generator,  # Optional: Use a separate generator for validation data
    validation_steps=10  # Optional: Specify the number of validation steps per epoch
)



