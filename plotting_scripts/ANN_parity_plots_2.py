#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:52:12 2023

@author: maximsecor
"""


import warnings
from sklearn.metrics import mean_absolute_error
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os.path import exists
import seaborn as sns
import sys

np.set_printoptions(precision=4,suppress=True)

#%%

file_features = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OOH/UHF_density_matrices_full.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OOH/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_WFT_DM_full')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')
plt.figure(figsize=(6,6))



plt.scatter(test_y,predictions_test,c = 'lightblue',edgecolor = 'blue',alpha=1,linewidth=0.5,s=35)
# plt.savefig("/Users/maximsecor/Desktop/parity_WFT_OOH.tif",dpi=800)
# plt.show()


file_features = '/Users/maximsecor/Desktop/MNC_NEW/UHF_O/UHF_density_matrices_full.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/UHF_O/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_WFT_O')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

plt.scatter(test_y,predictions_test,c = 'pink',edgecolor = 'red',alpha=1,linewidth=0.5,s=35)
# plt.savefig("/Users/maximsecor/Desktop/parity_WFT_O.tif",dpi=800)
# plt.show()


file_features = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OH/UHF_density_matrices_full.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OH/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_WFT_OH')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

# plt.xlim(0,180)
plt.xticks([-2,-1,0,1,2,3,4])


plt.scatter(test_y,predictions_test,c = 'lightgreen',edgecolor = 'green',alpha=1,linewidth=0.5,s=35)
plt.savefig("/Users/maximsecor/Desktop/parity_WFT_OH.tif",dpi=800)
plt.show()

#%%

file_features = '/Users/maximsecor/Desktop/MNC_NEW/UHF_H/UHF_density_matrices_full.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/UHF_H/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_WFT_H')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

plt.xlim(-1,3)
plt.ylim(-1,3)

plt.scatter(test_y,predictions_test,c = 'lightblue',edgecolor = 'blue',alpha=1,linewidth=0.5,s=35)
plt.savefig("/Users/maximsecor/Desktop/parity_WFT_H.tif",dpi=800)
plt.show()

#%%

file_features = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/Corr_64.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_C64')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

plt.xlim(0,4.5)
plt.ylim(0,4.5)

plt.scatter(test_y,predictions_test,c = 'blue',edgecolor = 'navy',alpha=0.5,linewidth=0.5,s=35)
# plt.savefig("/Users/maximsecor/Desktop/parity_C256.tif",dpi=800)
# plt.show()

file_features = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/PCA_256.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_P256')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

plt.xlim(0,4.5)
plt.ylim(0,4.5)

plt.scatter(test_y,predictions_test,c = 'red',edgecolor = 'darkred',alpha=0.5,linewidth=0.5,s=35)
# plt.savefig("/Users/maximsecor/Desktop/parity_P256.tif",dpi=800)
# plt.show()

file_features = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/RFE_64.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

train_X, val_X, train_y, val_y = train_test_split(features, target, test_size=(2/10), random_state=706)
val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=(1/2), random_state=706)

train_X_tf = tf.convert_to_tensor(train_X, np.float32)
train_y_tf = tf.convert_to_tensor(train_y, np.float32)
val_X_tf = tf.convert_to_tensor(val_X, np.float32)
val_y_tf = tf.convert_to_tensor(val_y, np.float32)
test_X_tf = tf.convert_to_tensor(test_X, np.float32)
test_y_tf = tf.convert_to_tensor(test_y, np.float32)

model = load_model('/Users/maximsecor/Desktop/MNC_NEW/SAVED_ANNS/saved_model_R64')

predictions_test = model.predict(test_X_tf)
MAE = mean_absolute_error(test_y_tf, predictions_test)
print('Test Set Error =', MAE)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5
plt.gca().set_aspect('equal')

plt.xlim(0,4.5)
plt.ylim(0,4.5)

plt.xticks([0,1,2,3,4])

plt.scatter(test_y,predictions_test,c = 'green',edgecolor = 'darkgreen',alpha=0.5,linewidth=0.5,s=35)
plt.savefig("/Users/maximsecor/Desktop/parity_R64.tif",dpi=800)
plt.show()

#%%




