# https://github.com/ayu-22/Deep-Steganography

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc
from tqdm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
import cv2
import shutil
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from scipy import ndimage
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder
from __future__ import print_function, division
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam


files = os.listdir('tiny-imagenet-200/train')
files_te = os.listdir('tiny-imagenet-200/test/images')


x_train = np.empty((2000,64,64,3), 'uint64')
a=0
for i in range(200):
  idd = np.random.randint(0, 500, 10)
  for j in range(10):
    image = cv2.imread('tiny-imagenet-200/train/'+files[i]+'/images/'+files[i]+'_'+str(idd[j])+'.JPEG')
    x_train[a] = image
    a=a+1
    
x_test = np.empty((2000,64,64,3), 'uint64')
a=0
for i in range(2000):
    image = cv2.imread('tiny-imagenet-200/test/images/'+files_te[i])
    x_test[a] = image
    a=a+1

input_S = x_train[0:1000]

input_C = x_train[1000:]

input_C = input_C/255.0
input_S = input_S/255.0

beta = 1.0
def rev_loss(true,pred):
  loss = beta*K.sum(K.square(true-pred))
  return loss
  
def full_loss(true,pred):
  message_true, container_true = true[...,0:3], true[...,3:6] 
  message_pred, container_pred = pred[...,0:3], pred[...,3:6] 
  
  message_loss = rev_loss(message_true, message_pred)
  container_loss = K.sum(K.square(container_true-container_pred))
  
  loss = message_loss + container_loss
  return loss

def prep_and_hide_network(input_size):
  input_message = Input(shape=(input_size))
  input_cover = Input(shape=(input_size))
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_message)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(input_message)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(input_message)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x = concatenate([input_cover,x])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (4,4), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (5,5), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  image_container = Conv2D(3, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  
  encoder = Model(inputs = [input_message, input_cover],
                  outputs = image_container)
  return encoder



def reveal_network(input_size, fixed=False):
  reveal_input = Input(shape=(input_size))
  
  input_with_noise = GaussianNoise(0.01)(reveal_input)
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_with_noise)
  x2 = Conv2D(10, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_with_noise)
  x3 = Conv2D(5, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_with_noise)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  x1 = Conv2D(50, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x2 = Conv2D(10, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x3 = Conv2D(5, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  x = concatenate([x1, x2, x3])
  
  message = Conv2D(3, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
  
  reveal =  Model(inputs = reveal_input,
                  outputs = message)
  
  return reveal

shape = input_S.shape[1:]

input_message = Input(shape = (shape))
input_container = Input(shape = (shape))

prep_and_hide = prep_and_hide_network(shape)
reveal = reveal_network(shape)

reveal.compile(optimizer = 'adam',
              loss = rev_loss)

reveal.trainable = False

output_container = prep_and_hide([input_message, input_container])
output_message = reveal(output_container)

deep_stean = Model(inputs = [input_message, input_container],
                outputs = concatenate([output_message, output_container]))

deep_stean.compile(optimizer = 'adam',
                  loss = full_loss)

def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003

m = input_S.shape[0]
loss_history = []
batch_size = 32
for epoch in range(1000):
    np.random.shuffle(input_S)
    np.random.shuffle(input_C)
    itera  = int(m/batch_size)
    f_loss_mean = 0
    r_loss_mean = 0
    for i in range(itera):
      batch_message = input_S[i*batch_size:min((i+1)*batch_size,m)]
      batch_cover = input_C[i*batch_size:min((i+1)*batch_size,m)]
      
      container = prep_and_hide.predict([batch_message,batch_cover])
      
      f_loss = deep_stean.train_on_batch(x = [batch_message,batch_cover],
                                         y = np.concatenate((batch_message,batch_cover), axis = 3))
      r_loss = reveal.train_on_batch(x = container,
                                    y = batch_message)
      
      f_loss_mean = f_loss_mean + f_loss
      r_loss_mean = r_loss_mean + r_loss
      
      print('Epoch = '+str(epoch)+' batch = '+str(i)+' | full loss = '+str(f_loss)+' | rev_loss = '+str(r_loss))
      
    f_loss_mean = f_loss_mean/itera
    r_loss_mean = r_loss_mean/itera
    print('Epoch = '+str(epoch)+' | mean full loss = '+str(f_loss_mean)+' | mean rev_loss = '+str(r_loss_mean))
    print('--------------------Epoch '+str(epoch)+' complete--------------------')  
    K.set_value(deep_stean.optimizer.lr, lr_schedule(epoch))
    K.set_value(reveal.optimizer.lr, lr_schedule(epoch))
