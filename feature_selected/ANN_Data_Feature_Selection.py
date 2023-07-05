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

file_features = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OOH/UHF_density_matrices_full.csv'
file_target = '/Users/maximsecor/Desktop/MNC_NEW/UHF_OOH/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

#%%

total = np.concatenate((target.reshape(-1,1),features),1)
df_total = pd.DataFrame(total)
pearson_corr = df_total.corr().values
corr_sort = np.argsort(np.abs(pearson_corr[1:,0]))

features_corr_16 =[]
features_corr_64 =[]
features_corr_256 =[]

for i in range(16):
    features_corr_16.append(features[:,corr_sort[::-1][i]])
for i in range(64):
    features_corr_64.append(features[:,corr_sort[::-1][i]])
for i in range(256):
    features_corr_256.append(features[:,corr_sort[::-1][i]])

features_corr_16 = np.array(features_corr_16).T
features_corr_64 = np.array(features_corr_64).T
features_corr_256 = np.array(features_corr_256).T

file_Corr_16 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/Corr_16.csv'
file_Corr_64 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/Corr_64.csv'
file_Corr_256 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/Corr_256.csv'

os.system('touch ' + file_Corr_16)
os.system('touch ' + file_Corr_64)
os.system('touch ' + file_Corr_256)

df_features_corr_16 = pd.DataFrame(features_corr_16)
df_features_corr_64 = pd.DataFrame(features_corr_64)
df_features_corr_256 = pd.DataFrame(features_corr_256)

df_features_corr_16.to_csv(file_Corr_16, index = False, header=True)
df_features_corr_64.to_csv(file_Corr_64, index = False, header=True)
df_features_corr_256.to_csv(file_Corr_256, index = False, header=True)


#%%

pca = PCA(n_components=256)
pca.fit(features)
features_pca = pca.transform(features)

features_pca_16 = features_pca[:,:16]
features_pca_64 = features_pca[:,:64]
features_pca_256 = features_pca

file_PCA_16 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/PCA_16.csv'
file_PCA_64 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/PCA_64.csv'
file_PCA_256 = '/Users/maximsecor/Desktop/MNC_NEW/Feature_Selected/PCA_256.csv'

os.system('touch ' + file_PCA_16)
os.system('touch ' + file_PCA_64)
os.system('touch ' + file_PCA_256)

df_features_pca_16 = pd.DataFrame(features_pca_16)
df_features_pca_64 = pd.DataFrame(features_pca_64)
df_features_pca_256 = pd.DataFrame(features_pca_256)

df_features_pca_16.to_csv(file_PCA_16, index = False, header=True)
df_features_pca_64.to_csv(file_PCA_64, index = False, header=True)
df_features_pca_256.to_csv(file_PCA_256, index = False, header=True)

%%

print(pca.get_covariance())
print(features_pca)

#%%

# print(pearson_corr[1:,0][corr_sort[::-1][:16]])
print(matrix_orb_list[corr_sort])
plt.hist(pearson_corr[1:,0],bins=50)

#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if (temp_orbs[1]) == (temp_orbs[2]):
        print(temp_orbs)
        temp_1.append(pearson_corr[1:,0][corr_sort[i]])
    else:
        temp_2.append(pearson_corr[1:,0][corr_sort[i]])        
    
sns.kdeplot(temp_1,cut=1,color = 'k')
sns.kdeplot(temp_2,cut=1)

#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if (temp_orbs[1][1]) == (temp_orbs[2][1]):
        print(temp_orbs)
        temp_1.append(pearson_corr[1:,0][corr_sort[i]])
    else:
        temp_2.append(pearson_corr[1:,0][corr_sort[i]])        
    
sns.kdeplot(temp_1,cut=1,color = 'k')
sns.kdeplot(temp_2,cut=1)

#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if temp_orbs[1][2] in ['4s','4p','3d']:
        print(temp_orbs)
        temp_1.append(pearson_corr[1:,0][corr_sort[i]])
    else:
        temp_2.append(pearson_corr[1:,0][corr_sort[i]])        
    
sns.kdeplot(temp_1,cut=1,color = 'k')
sns.kdeplot(temp_2,cut=1)

#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if temp_orbs[1][2] in ['4s','4p','3d']:
        print(temp_orbs)
        temp_1.append(pearson_corr[1:,0][corr_sort[i]])
    else:
        temp_2.append(pearson_corr[1:,0][corr_sort[i]])        
    
sns.kdeplot(temp_1,cut=1,color = 'k')
sns.kdeplot(temp_2,cut=1)


