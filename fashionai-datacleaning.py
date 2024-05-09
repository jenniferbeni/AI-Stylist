#data pre-processing and feature engineering 

import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

#normalize
train_images = train_images/255.0
test_images = test_images/255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#feature engineering

def is_cropped_top(train_images):
        bottom_half = train_images[train_images.shape[0] // 2:, :]
        average_intensity_bottom_half = np.mean(bottom_half)
        return average_intensity_bottom_half < 100


class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def assign_occasion(label):
  if label in [0,1,3,6,7]:
      return 0 #casual
  elif label in [4,5,8]:
    return 1 #formal
  else: 
      return 2 #party


train_occasion_labels = np.array([assign_occasion(label) for label in train_labels])

test_occasion_labels = np.array([assign_occasion(label) for label in test_labels])

occasion_names = ['Casual', 'Formal', 'Party']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(occasion_names[train_occasion_labels[i]])
plt.show()

#one-hot encoding
#train_occasion_labels = tf.keras.utils.to_categorical(train_occasion_labels)
#test_occasion_labels = tf.keras.utils.to_categorical(test_occasion_labels)


#define model 
image_input_shape = (28,28,1)
image_input = input(shape = image_input_shape, name = 'image_input')

#text input 
max_sequence_length = 20
vocab_size = 10000
embedding_dim = 100
text_input = input(shape=max_sequence_length, name='text input')

# Save preprocessed images and labels
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
np.save('train_occasion_labels.npy', train_occasion_labels)
np.save('test_occasion_labels.npy', test_occasion_labels)