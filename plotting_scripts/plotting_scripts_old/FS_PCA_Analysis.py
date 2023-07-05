#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:32:00 2022

@author: maximsecor
"""

import warnings
import time

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import exists
import seaborn as sns
import sys

np.set_printoptions(precision=4,suppress=True)

#%%

file_features = '/Users/maximsecor/Desktop/MNC/Cleaned_ANN_input_Data/OOH_UHF/UHF_density_matrices_full.csv'
df_features = pd.read_csv(file_features)
features = df_features.values

pca = PCA(n_components=3)
pca.fit(features)
features_pca = pca.transform(features)

#%%

# print(pca.get_covariance())
print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
plt.scatter(features_pca[:,0],features_pca[:,1])

#%%

inverse = (pca.inverse_transform(np.array([[1,0,0],[0,1,0],[0,0,1]])))

#%%

# plt.xlim(0.75,1.25)
# plt.ylim(0,0.1)
sns.kdeplot(inverse[0])
sns.kdeplot(inverse[1])
# sns.kdeplot(inverse[2])

#%%

for i in range(16):
    print(np.argsort(inverse[0])[::-1][i],inverse[0,np.argsort(inverse[0])[::-1][i]],matrix_orb_list[np.argsort(inverse[0])[::-1][i]])

#%%

for i in range(16):
    print(np.argsort(inverse[1])[::-1][i],inverse[1,np.argsort(inverse[1])[::-1][i]],matrix_orb_list[np.argsort(inverse[1])[::-1][i]])

#%%

for i in range(16):
    print(np.argsort(inverse[2])[::-1][i],inverse[2,np.argsort(inverse[2])[::-1][i]],matrix_orb_list[np.argsort(inverse[2])[::-1][i]])


#%%

plt.scatter(features[:,742],features_pca[:,0])




