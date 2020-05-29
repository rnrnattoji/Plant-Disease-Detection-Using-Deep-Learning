'''Preprocessing and Data Formation'''

# Library imports for preprocessing and data formation
  
import os                
# For using the percentage bar
from tqdm import tqdm  
# For shuffling the training data
from random import shuffle
# For dealing with arrays   
import numpy as np
# Loading parameter values   
from Parameters import *
# Using opencv for resizing the images to 50X50
import cv2  


def get_leaf_actual_label(leaf_image):

    '''
        Supervised Classification Problem
        Labelled Data
        The starting letter of each leaf image file name indicates if its healthy/unhealthy(bacterial disease, viral disease, late blight disease)
    '''

    # Getting the character representation of the label
    actual_label = leaf_image[0]
  
    # Featurization/Vectorization : Forming a numeric representation of the label for the deep learning model 
    # Healthy
    if actual_label == 'h': 
        return [1,0,0,0] 
    # Yellow Leaf Curl Virus  
    elif actual_label == 'v': 
        return [0,1,0,0]
    # Late Blight Disease    
    elif actual_label == 'l': 
        return [0,0,1,0]
    # Bacterial spot disease     
    elif actual_label == 'b': 
        return [0,0,0,1]


def build_data(DATA):
    '''
        Builds data(training/testing) for the deep learning model
        Iterates through the images in the data directory
        Reads them using opencv library
        Finally appends them to data
    '''
    data = []

    for leaf_image in tqdm(os.listdir(DATA)):
        # Getting actual label of the leaf
        leaf_actual_label = get_leaf_actual_label(leaf_image)
        # Forming leaf image path
        leaf_image_path = os.path.join(DATA, leaf_image)
        # Reading leaf color image from its path using OpenCV
        leaf_image = cv2.imread(leaf_image_path, cv2.IMREAD_COLOR)
        # Required image size - 50X50
        required_size = (IMAGE_SIZE, IMAGE_SIZE)
        # Resizing the image to required size
        resized_image = cv2.resize(leaf_image, required_size)
        # Appending image along with its label to data
        data.append([np.array(resized_image), np.array(leaf_actual_label)])
    
    # Shuffling data
    shuffle(data)

    return data


'''Training Data'''
# Check if training data already exists
try:
    training_data = np.load('training_data.npy')
# If not, build training data and save it for reuse   
except:     
    training_data = build_data(TRAINING_DATA)
    np.save('training_data.npy', training_data)


'''Testing Data'''
# Check if testing data already exists
try:
    testing_data = np.load('testing_data.npy')
# If not, build testing data and save it for reuse   
except:     
    testing_data = build_data(TESTING_DATA)
    np.save('testing_data.npy', testing_data)    


''' Deep Learning using Tensorflow '''

#Library imports for deep learning using tensorflow

import tensorflow as tf
import tflearn
# CNN layers
from tflearn.layers.conv import conv_2d, max_pool_2d
# Core Layers
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

'''
    Neuron Activation Function - relu 
    Gradient Descent Optimizer - Adam
    Loss Function - Categorical Cross-Entropy 

'''

tf.reset_default_graph()

# Add layers and form the neural network
cnn = input_data(name='input', shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])

cnn = conv_2d(cnn, 32, 3, activation='relu')
cnn = max_pool_2d(cnn, 3)

cnn = conv_2d(cnn, 64, 3, activation='relu')
cnn = max_pool_2d(cnn, 3)

cnn = conv_2d(cnn, 128, 3, activation='relu')
cnn = max_pool_2d(cnn, 3)

cnn = conv_2d(cnn, 32, 3, activation='relu')
cnn = max_pool_2d(cnn, 3)

cnn = conv_2d(cnn, 64, 3, activation='relu')
cnn = max_pool_2d(cnn, 3)

cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.8)

cnn = fully_connected(cnn, 4, activation='softmax')
cnn = regression(cnn, loss='categorical_crossentropy', name='targets', learning_rate=LEARNING_RATE, optimizer='adam')

# Deep Learning Model
dl_model = tflearn.DNN(cnn)

# Check if model already exists, load it
if os.path.exists('{}.meta'.format(DEEP_LEARNING_MODEL)):
    dl_model.load(DEEP_LEARNING_MODEL)
    print('MODEL LOADED!')
# Else, build and save the model    
else:
    # Total images = 4000
    # Healthy = 1000
    # YLCV = 1000
    # BS = 1000
    # LB = 1000
    # Use 3000 for training and 1000 for validation

    train = training_data[:-1000]
    test = training_data[-1000:]
    X_train = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
    y_train = [i[1] for i in train]
    X_test = np.array([i[0] for i in test]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3)
    y_test = [i[1] for i in test]

    # Fit training data to model, also provide validation set
    dl_model.fit({'input': X_train}, {'targets': y_train}, n_epoch=8, validation_set=({'input': X_test}, {'targets': y_test}),
        snapshot_step=40, show_metric=True, run_id=DEEP_LEARNING_MODEL)

    # Save the model
    dl_model.save(DEEP_LEARNING_MODEL)


# Testing Accuracy

def get_index(data):

    if data[0] == 1:
        return 0
    elif data[1] == 1:
        return 1
    elif data[2] == 1:
        return 2
    elif data[3] == 1:
        return 3


correct = 0

for test_point in testing_data:

    actual_label = test_point[1]
    predicted_label = np.argmax(dl_model.predict([test_point[0]])[0])
    actual_label = get_index(actual_label)

    if actual_label == predicted_label:
        correct += 1

print(correct, "out of", len(testing_data), "testing points classified correctly")

print("Testing accuracy : " + str(correct/len(testing_data)*100))







        
