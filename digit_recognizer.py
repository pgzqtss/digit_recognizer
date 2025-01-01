import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from skimage.transform import resize
import cv2
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
''' Keras: an open-source deep learning library written in Python
    Provides high-level API for building and training neural networks'''
import keras
from keras import layers
import sys

# Set default enciding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# for Matplotlib to display inline
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma') # set colormap for images
warnings.filterwarnings("ignore") # clean up output cells



# <1> Loading data and extracting -----------------------------------------------

# load data to panda dataframe
train = pd.read_csv('train.csv')
# select first 5000 rows for training
train = train.loc[:5000]

# split data in dataframe into features and labels
def Extraction(df):
    # Select label column from dataframe
    label = df['label']
    # drop label column and assigns remaining columns to features
    feature = df.drop(['label'], axis=1) # axis=0 rows, axis=1 columns
    return feature, label

# Extract the subset dataset into features and labels
Feature, Label = Extraction(train)



# <2> Prepare data for training neural network -------------------------------------

# Normalize the features to each image for training neural network
image_data = []
for image_row in tqdm(train.index): # tqdm trancks the progress
    tmp   = np.array(list(Feature.loc[image_row])) # convert row to numpy array
    tmp   = np.resize(tmp,(28,28)) # resize 1D array to 2D array
    img_r = resize(tmp, (128, 128, 1)) # resize 2D array to 128x128x1
    image = tf.image.convert_image_dtype(img_r, dtype=tf.float32) # convert image to TensorFlow format
    image_data.append(img_r)
    
# convert the labels into a format for training neural network
train_labels = to_categorical(Label, 10) # 10 classes, 0-9
# converts a class vector (integers) to binary class matrix (one-hot encoding)



# print plt
plt.figure(figsize=(10,10))
for i in range(16):
    image = image_data[i]
    plt.subplot(4, 4, i+1)
    plt.imshow(image)
    plt.axis('off')
plt.show()



''' By using InceptionV3 model, it expects input imahes
    with 3 chanels (RGB). Since dataset images are in grayscale
    (1 channel), below code converts them.'''
# Print the shape of X_train_norm
print(np.array(image_data).shape[1])
# Load InceptionV3 model
inception_tl_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(128, 128, 3),
    pooling=None,
    classes=10
)


# --------------------------------------------------------------------------------------------
# Defining model ---------------------------------------------------------------------
# Convolutional Neural Network(CNN) model using Keras' Sequential API
model = keras.Sequential([
    # Convolutional base
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[128, 128, 1]),
    # Max pooling layer
    layers.MaxPool2D(),
    
    # Repeats
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3,activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    # Flattens the 3D output of concolutional layers into 1D vector
    layers.Flatten(),
    
    
    # Dense layers/ fully connected layers
    layers.Dense(128,activation="relu"), 
    layers.Dense(64,activation="relu"),
    layers.Dense(32,activation="relu"),
    # Output layer (0-9)
    layers.Dense(10,activation="softmax")
])

'''
To conclude:
Convolutional layers: detect patterns, learn features
Pooling layers: reduce spatial dimensions, make network more robust
Dense layers: decision making, classifying image to specific category
'''

# Compiling model ---------------------------------------------------------------------
# Define optimizer: Adam Optimizer, epsolon is a small constant added to the denominator to avoid division by zero
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
# Compile model
model.compile(
    optimizer=optimizer,
    # Loss function
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training model ---------------------------------------------------------------------
'''
np.array(image_data): convert list of images to numpy array
np.array(train_labels): convert list of labels to numpy array
epochs: number of times the model will train on the entire dataset
verbose=True: training progress will be displayed on console
'''
model.fit(np.array(image_data), np.array(train_labels), epochs=10, verbose=True)

# Evaluate model ---------------------------------------------------------------------
# Evaluate method computes the loss -> returns loss value and metric values
model.evaluate(np.array(image_data), np.array(train_labels))

# --------------------------------------------------------------------------------------------
# Read the training and test dataset and load into panda dataframe
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Separate the features and labels
X_train,Y_train = train.drop("label",axis=1),train[['label']]
# Normalizing features: range(0-255) -> range(0-1)
X_train_norm = X_train/255.0
X_test = test/255.0
# One-hot encoding the labels
y_train = to_categorical(Y_train)
'''
One-hot encoding
- transforms the integer class labels into binary matrix representation
- each class label is converted into a vector
- class labels set to 1 and rest to 0
- e.g. 10 classes(0-9), so class 3 -> [0,0,0,1,0,0,0,0,0,0]
- WHY?
    - output probability distribution over classes
    - compute loss and gradients
'''
print(np.array(X_train_norm).shape[1])

# Call the set model and train with dataset
model = keras.Sequential()
model.add(layers.Dense(512, input_dim=np.array(X_train_norm).shape[1], activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Normalize data
model.fit(X_train_norm, y_train, epochs=10, batch_size=10)

# --------------------------------------------------------------------------------------------
# Take test data and pass through trained NN model
p_data = model.predict(X_test)
p_data

# Generate IDs and corresponding predicted labels
i=1
ImageId = []
Label   = []
for item in p_data:
    ImageId.append(i)
    Label.append(np.argmax(item)) 
    i+=1

# Create a dictionary with ImageId and Label
tmp = {"ImageId":ImageId,"Label":Label}

# Convert dictionary to panda dataframe
df = pd.DataFrame(tmp)

# Save the dataframe to a CSV file
df = df.set_index("ImageId")
df.to_csv("out.csv",header=True)

# Save the model
model.save('model.h5')