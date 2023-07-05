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
file_target = '/Users/maximsecor/Desktop/MNC/Cleaned_ANN_input_Data/OOH_UHF/cata_binding.csv'

df_features = pd.read_csv(file_features)
df_target = pd.read_csv(file_target)

features = df_features.values
target = df_target.values

total = np.concatenate((target.reshape(-1,1),features),1)
df_total = pd.DataFrame(total)
pearson_corr = df_total.corr().values
corr_sort = np.argsort(np.abs(pearson_corr[1:,0]))

#%%

orbital_list = [(35, 'Fe', '1s', ''), (35, 'Fe', '2s', ''), (35, 'Fe', '3s', ''), (35, 'Fe', '4s', ''), (35, 'Fe', '2p', 'x'), (35, 'Fe', '2p', 'y'), (35, 'Fe', '2p', 'z'), (35, 'Fe', '3p', 'x'), (35, 'Fe', '3p', 'y'), (35, 'Fe', '3p', 'z'), (35, 'Fe', '4p', 'x'), (35, 'Fe', '4p', 'y'), (35, 'Fe', '4p', 'z'), (35, 'Fe', '3d', 'xy'), (35, 'Fe', '3d', 'yz'), (35, 'Fe', '3d', 'z^2'), (35, 'Fe', '3d', 'xz'), (35, 'Fe', '3d', 'x2-y2'),
                (36, 'N', '1s', ''), (36, 'N', '2s', ''), (36, 'N', '2p', 'x'), (36, 'N', '2p', 'y'), (36, 'N', '2p', 'z'), (37, 'N', '1s', ''), (37, 'N', '2s', ''), (37, 'N', '2p', 'x'), (37, 'N', '2p', 'y'), (37, 'N', '2p', 'z'), (38, 'N', '1s', ''), (38, 'N', '2s', ''), (38, 'N', '2p', 'x'), (38, 'N', '2p', 'y'), (38, 'N', '2p', 'z'), (39, 'N', '1s', ''), (39, 'N', '2s', ''), (39, 'N', '2p', 'x'), (39, 'N', '2p', 'y'), (39, 'N', '2p', 'z')]

matrix_orb_list = []
matrix_3d_list = []

q1 = 0
q2 = 0
for i in range(len(orbital_list)):
    for j in range(len(orbital_list)):
        if i>=j:
            matrix_orb_list.append(['alpha',orbital_list[i],orbital_list[j]])
            if orbital_list[i][2] == '3d':
                if orbital_list[j][2] == '3d':
                    matrix_3d_list.append([orbital_list[i][3],orbital_list[j][3]])
                    q1 += 1
            q2 += 1

q1 = 0
q2 = 0
for i in range(len(orbital_list)):
    for j in range(len(orbital_list)):
        if i>=j:
            matrix_orb_list.append(['beta',orbital_list[i],orbital_list[j]])
            if orbital_list[i][2] == '3d':
                if orbital_list[j][2] == '3d':
                    matrix_3d_list.append([orbital_list[i][3],orbital_list[j][3]])
                    q1 += 1
            q2 += 1

matrix_orb_list = np.array(matrix_orb_list)
matrix_3d_list = np.array(matrix_3d_list)

#%%

alpha_corr_list = []
beta_corr_list = []

for i in range(int(len(matrix_orb_list)/2)):
    
    alpha_corr = pearson_corr[1:,0][np.where(corr_sort==i)[0][0]]
    beta_corr = pearson_corr[1:,0][np.where(corr_sort==i+741)[0][0]]

    alpha_corr_list.append(alpha_corr)
    beta_corr_list.append(beta_corr)

print(alpha_corr_list)

#%%

temp_1 = []
temp_2 = []
temp_3 = []

for i in range(len(alpha_corr_list)):
    temp_orbs = matrix_orb_list[i]
    if temp_orbs[1] == temp_orbs[1]:
        print(temp_orbs)
        temp_1.append(alpha_corr_list[i])
        temp_2.append(beta_corr_list[i])
        temp_3.append(np.abs(alpha_corr_list[i])+np.abs(beta_corr_list[i]))
    else:
        temp_2.append(beta_corr_list[i])
    
sns.kdeplot(temp_1,cut=1,color = 'k')
# sns.kdeplot(temp_3,cut=2,color = 'r')
sns.kdeplot(temp_2,cut=1)

#%%

# print(pearson_corr[1:,0][corr_sort[::-1][:16]])
for i in range(64):
    print(i,matrix_orb_list[corr_sort[::-1]][i])
# plt.hist(pearson_corr[1:,0],bins=50)


#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if temp_orbs[1] == temp_orbs[2]:
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
        temp_1.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))
    else:
        temp_2.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))        

plt.xlim(0,1)
sns.kdeplot(temp_1,cut=15,color = 'k')
sns.kdeplot(temp_2,cut=15)

#%%

temp_1 = []
temp_2 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if temp_orbs[0] == 'alpha':
        print(temp_orbs)
        temp_1.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))
    else:
        temp_2.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))     

plt.xlim(0,1)    
sns.kdeplot(temp_1,color = 'k')
sns.kdeplot(temp_2)

print(np.mean(temp_1))
print(np.mean(temp_2))

#%%

temp_1 = []
temp_2 = []
temp_3 = []

for i in range(len(corr_sort)):
    temp_orbs = (matrix_orb_list[corr_sort[i]])
    if temp_orbs[1][1] == 'Fe' and temp_orbs[2][1] == 'Fe':
        temp_1.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))
    if temp_orbs[1][1] == 'N' and temp_orbs[2][1] == 'N':
        temp_2.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))        
    else:
        # print(temp_orbs)
        temp_3.append(np.abs(pearson_corr[1:,0][corr_sort[i]]))       
    
plt.xlim(0.6,1) 
sns.kdeplot(temp_1,color = 'k')
sns.kdeplot(temp_2)
sns.kdeplot(temp_3)

print(np.mean(temp_1))
print(np.mean(temp_2))
print(np.mean(temp_3))

#%%

for i in range(5):
    print(temp_1[::-1][i],matrix_orb_list[np.where(np.abs(pearson_corr[1:,0])==temp_1[::-1][i])[0][0]])
    
for i in range(5):
    print(temp_2[::-1][i],matrix_orb_list[np.where(np.abs(pearson_corr[1:,0])==temp_2[::-1][i])[0][0]])
    
for i in range(5):
    print(temp_3[::-1][i],matrix_orb_list[np.where(np.abs(pearson_corr[1:,0])==temp_3[::-1][i])[0][0]])

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

plt.xlim(-1,1)
sns.kdeplot(temp_1,cut=1,color = 'k')
sns.kdeplot(temp_2,cut=1)

print(np.mean(np.abs(temp_1)))
print(np.mean(np.abs(temp_2)))

#%%

print(np.max(temp_1))
print(np.min(temp_1))

print(np.max(temp_2))
print(np.min(temp_2))

#%%

print([np.where(pearson_corr[1:,0]==np.max(temp_1))[0][0]])
print([np.where(pearson_corr[1:,0]==np.min(temp_1))[0][0]])

print([np.where(pearson_corr[1:,0]==np.max(temp_2))[0][0]])
print([np.where(pearson_corr[1:,0]==np.min(temp_2))[0][0]])

#%%

print(matrix_orb_list[np.where(pearson_corr[1:,0]==np.max(temp_1))[0][0]])
print(matrix_orb_list[np.where(pearson_corr[1:,0]==np.min(temp_1))[0][0]])

print(matrix_orb_list[np.where(pearson_corr[1:,0]==np.max(temp_2))[0][0]])
print(matrix_orb_list[np.where(pearson_corr[1:,0]==np.min(temp_2))[0][0]])

#%%

plt.scatter(features[:,152],target)







