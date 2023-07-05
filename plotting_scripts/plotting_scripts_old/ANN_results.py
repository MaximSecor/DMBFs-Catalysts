#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:56:34 2022

@author: maximsecor
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time as time
import os
import sys
import seaborn as sns

from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

np.set_printoptions(precision=4,suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#%%

file_features = '/Users/maximsecor/Desktop/MNC/Cleaned_ANN_input_Data/Feature_Selected/Corr_16.csv'
file_target = '/Users/maximsecor/Desktop/MNC/Cleaned_ANN_input_Data/OOH_UHF/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

print(np.min(target))
print(np.max(target))

#%%

model = load_model('/Users/maximsecor/Desktop/MNC/ANN_models/saved_model_Corr')

#%%

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

#%%

# plt.scatter(test_y_tf,predictions_test)

plt.scatter(test_y_tf,predictions_test,c = "red",edgecolor = "darkred",alpha=1,linewidth=0.75,s=16)
plt.savefig("/Users/maximsecor/Desktop/model_Corr_16.tif",dpi=800)







