#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:32:00 2022

@author: maximsecor
"""

import warnings
import time
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from keras.optimizers import Adam
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import exists
import seaborn as sns
import sys

import matplotlib as mpl

np.set_printoptions(precision=4,suppress=True)

#%%

def data(a1,a2):
    
    if a1 == 0:
        temp_1 = 'UHF_DOPED'
        temp_2 = 'UHF_METAL'
    else:
        temp_1 = 'B3LYP_DOPED'
        temp_2 = 'B3LYP_METAL'
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/FreeEnergy.csv'
    MNC_data = pd.read_csv(file_MNC)
    MNC_data = MNC_data.values
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/FreeEnergy_NoDopant.csv'
    MNC_data_NoDopant = pd.read_csv(file_MNC)
    MNC_data_NoDopant = MNC_data_NoDopant.values
    
    MNC_data_reshaped = MNC_data.reshape(20, 5, 102)
    MNC_data_reshaped = (np.moveaxis(MNC_data_reshaped, 0, -1).T.reshape(2040, 5))
    
    MNC_data_NoDopant_reshaped = MNC_data_NoDopant.reshape(2, 5, 10)
    MNC_data_NoDopant_reshaped = (np.moveaxis(MNC_data_NoDopant_reshaped, 0, -1).T.reshape(20, 5))
    
    MNC_data_reshaped = np.concatenate((MNC_data_reshaped,MNC_data_NoDopant_reshaped))
    
    metal_list = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    dope_list = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 3, 3]

    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/DensityMatrices.csv'
    density_matrices_data = pd.read_csv(file_MNC)
    density_matrices_data = density_matrices_data.values
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/DensityMatrices.csv'
    density_matrices_data_NoDopant = pd.read_csv(file_MNC)
    density_matrices_data_NoDopant = density_matrices_data_NoDopant.values
    
    density_matrices_data_NoDopant_reorder = []
    for i in range(10):
        density_matrices_data_NoDopant_reorder.append(density_matrices_data_NoDopant[i*2])
    for i in range(10):
        density_matrices_data_NoDopant_reorder.append(density_matrices_data_NoDopant[i*2+1])
    density_matrices_data_NoDopant_reorder = np.array(density_matrices_data_NoDopant_reorder)
    
    density_matrices_full = np.concatenate((density_matrices_data,density_matrices_data_NoDopant_reorder))
    
    orbital_list = [(35, 'Fe', '1s', ''), (35, 'Fe', '2s', ''), (35, 'Fe', '3s', ''), (35, 'Fe', '4s', ''), (35, 'Fe', '2p', 'x'), (35, 'Fe', '2p', 'y'), (35, 'Fe', '2p', 'z'), (35, 'Fe', '3p', 'x'), (35, 'Fe', '3p', 'y'), (35, 'Fe', '3p', 'z'), (35, 'Fe', '4p', 'x'), (35, 'Fe', '4p', 'y'), (35, 'Fe', '4p', 'z'), (35, 'Fe', '3d', 'xy'), (35, 'Fe', '3d', 'yz'), (35, 'Fe', '3d', 'z^2'), (35, 'Fe', '3d', 'xz'), (35, 'Fe', '3d', 'x2-y2'),
                    (36, 'N', '1s', ''), (36, 'N', '2s', ''), (36, 'N', '2p', 'x'), (36, 'N', '2p', 'y'), (36, 'N', '2p', 'z'), (37, 'N', '1s', ''), (37, 'N', '2s', ''), (37, 'N', '2p', 'x'), (37, 'N', '2p', 'y'), (37, 'N', '2p', 'z'), (38, 'N', '1s', ''), (38, 'N', '2s', ''), (38, 'N', '2p', 'x'), (38, 'N', '2p', 'y'), (38, 'N', '2p', 'z'), (39, 'N', '1s', ''), (39, 'N', '2s', ''), (39, 'N', '2p', 'x'), (39, 'N', '2p', 'y'), (39, 'N', '2p', 'z')]
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/charge_Fe_list.csv'
    temp_data_1 = pd.read_csv(file_MNC)
    temp_data_1 = temp_data_1.values
    temp_data_1 = temp_data_1.reshape(2040, 2, 3)
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/charge_Fe_list.csv'
    temp_data_2 = pd.read_csv(file_MNC)
    temp_data_2 = temp_data_2.values
    temp_data_2 = temp_data_2.reshape(20, 2, 3)
    charge_Fe_list = np.concatenate((temp_data_1,temp_data_2))
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/charge_N4_list.csv'
    temp_data_1 = pd.read_csv(file_MNC)
    temp_data_1 = temp_data_1.values
    temp_data_1 = temp_data_1.reshape(2040, 8, 3)
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/charge_N4_list.csv'
    temp_data_2 = pd.read_csv(file_MNC)
    temp_data_2 = temp_data_2.values
    temp_data_2 = temp_data_2.reshape(20, 8, 3)
    charge_N4_list = np.concatenate((temp_data_1,temp_data_2))
    
    charge_1 = charge_Fe_list[:,0,2].reshape(-1,1)
    charge_2 = charge_N4_list[:,:4,2]
    charge_list_mulliken = np.concatenate((charge_2,charge_1),1)
    alpha_1 = charge_Fe_list[:,0,0].reshape(-1,1)
    alpha_2 = charge_N4_list[:,:4,0]
    alpha_list_mulliken = np.concatenate((alpha_2,alpha_1),1)
    beta_1 = charge_Fe_list[:,0,1].reshape(-1,1)
    beta_2 = charge_N4_list[:,:4,1]
    beta_list_mulliken = np.concatenate((beta_2,beta_1),1)
    
    charge_1 = charge_Fe_list[:,1,2].reshape(-1,1)
    charge_2 = charge_N4_list[:,4:,2]
    charge_list_lowdin = np.concatenate((charge_2,charge_1),1)
    alpha_1 = charge_Fe_list[:,1,0].reshape(-1,1)
    alpha_2 = charge_N4_list[:,4:,0]
    alpha_list_lowdin = np.concatenate((alpha_2,alpha_1),1)
    beta_1 = charge_Fe_list[:,1,1].reshape(-1,1)
    beta_2 = charge_N4_list[:,4:,1]
    beta_list_lowdin = np.concatenate((beta_2,beta_1),1)
    
    mulliken_atomic = np.concatenate((alpha_list_mulliken,beta_list_mulliken),1)
    lowdin_atomic = np.concatenate((alpha_list_lowdin,beta_list_lowdin),1)
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/pop_Fe_list.csv'
    temp_data_1 = pd.read_csv(file_MNC)
    temp_data_1 = temp_data_1.values
    temp_data_1 = temp_data_1.reshape(2040, 36, 2)
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/pop_Fe_list.csv'
    temp_data_2 = pd.read_csv(file_MNC)
    temp_data_2 = temp_data_2.values
    temp_data_2 = temp_data_2.reshape(20, 36, 2)
    
    pop_Fe_list = np.concatenate((temp_data_1,temp_data_2))
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_1+'/pop_N4_list.csv'
    temp_data_1 = pd.read_csv(file_MNC)
    temp_data_1 = temp_data_1.values
    temp_data_1 = temp_data_1.reshape(2040, 40, 2)
    
    file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/'+temp_2+'/pop_N4_list.csv'
    temp_data_2 = pd.read_csv(file_MNC)
    temp_data_2 = temp_data_2.values
    temp_data_2 = temp_data_2.reshape(20, 40, 2)
    
    pop_N4_list = np.concatenate((temp_data_1,temp_data_2))
    
    pop_alpha_1 = pop_Fe_list[:,:18,0]
    pop_alpha_2 = pop_N4_list[:,:20,0]
    pop_alpha_mulliken = np.concatenate((pop_alpha_2,pop_alpha_1[:,::-1]),1)
    pop_beta_1 = pop_Fe_list[:,:18,1]
    pop_beta_2 = pop_N4_list[:,:20,1]
    pop_beta_mulliken = np.concatenate((pop_beta_2,pop_beta_1[:,::-1]),1)
    
    pop_alpha_1 = pop_Fe_list[:,18:,0]
    pop_alpha_2 = pop_N4_list[:,20:,0]
    pop_alpha_lowdin = np.concatenate((pop_alpha_2,pop_alpha_1[:,::-1]),1)
    pop_beta_1 = pop_Fe_list[:,18:,1]
    pop_beta_2 = pop_N4_list[:,20:,1]
    pop_beta_lowdin = np.concatenate((pop_beta_2,pop_beta_1[:,::-1]),1)
    
    mulliken_orbital_full = np.concatenate((pop_alpha_mulliken,pop_beta_mulliken),1)
    lowdin_orbital_full = np.concatenate((pop_alpha_lowdin,pop_beta_lowdin),1)

    mulliken_orbital_3d = np.zeros((2060,10))
    lowdin_orbital_3d = np.zeros((2060,10))
    
    q1 = 0
    for i in range(len(orbital_list)):
        if orbital_list[i][2] == '3d':
            # print(i,orbital_list[i])
            for j in range(len(mulliken_orbital_full)):
                mulliken_orbital_3d[j,q1] =  mulliken_orbital_full[j,q1]
                mulliken_orbital_3d[j,q1+5] =  mulliken_orbital_full[j,q1+5]
                lowdin_orbital_3d[j,q1] =  lowdin_orbital_full[j,q1]
                lowdin_orbital_3d[j,q1+5] =  lowdin_orbital_full[j,q1+5]
            q1 += 1    
    
    density_matrices_3d = np.zeros((2060,30))
    matrix_orb_list = []
    
    q1 = 0
    q2 = 0
    for i in range(len(orbital_list)):
        for j in range(len(orbital_list)):
            if i>=j:
                matrix_orb_list.append(['alpha',orbital_list[i],orbital_list[j]])
                if orbital_list[i][2] == '3d':
                    if orbital_list[j][2] == '3d':
                        # print(orbital_list[i],orbital_list[j])
                        for i2 in range(2060):
                            density_matrices_3d[i2,q1] = density_matrices_full[i2,q2]
                            density_matrices_3d[i2,q1+15] = density_matrices_full[i2,q2+741]
                        q1 += 1
                q2 += 1
    
    
    gh2 = -1.18
    gw = -76.41724
    
    metal_list = [0,1,2,3,4,5,6,7,8,9]
    dope_list = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 3, 3]
    
    type_list_1 = []
    for i in range(len(MNC_data_reshaped)):
        if i < 2040:
            if (i%204) < 102:
                type_list_1.append(dope_list[i%204])
            else:
                type_list_1.append(dope_list[i%204-102]+4)
        else:
            if (i-2040) < 10:
                type_list_1.append(0)
            else:
                type_list_1.append(4)
                
    type_list_2 = []
    for i in range(len(MNC_data_reshaped)):
        if i < 2040:
            type_list_2.append(int(np.floor(i/204)))
        else:
            if (i-2040) < 10:
                type_list_2.append(i-2040)
            else:
                type_list_2.append(i-2050)

    cata_binding = []
    cata_ID = []
    if a2 == 0:
        for i in range(MNC_data_reshaped.shape[0]):
            if (MNC_data_reshaped[i][0]) != 0:
                if (MNC_data_reshaped[i][1]) != 0:                    
                    Free_star = MNC_data_reshaped[i][0]
                    Free_OOH = MNC_data_reshaped[i][1]                    
                    single_cata = ((Free_OOH+(3/2)*gh2)-(Free_star+2*gw))*27.11
                    if single_cata < 10:
                        if single_cata > -10:
                            cata_binding.append(single_cata)
                            cata_ID.append(i)  
    if a2 == 1:
        for i in range(MNC_data_reshaped.shape[0]):
            if (MNC_data_reshaped[i][0]) != 0:
                if (MNC_data_reshaped[i][2]) != 0:                    
                    Free_star = MNC_data_reshaped[i][0]
                    Free_O = MNC_data_reshaped[i][2]                     
                    single_cata = ((Free_O+gh2)-(Free_star+gw))*27.11
                    if single_cata < 10:
                        if single_cata > -10:
                            cata_binding.append(single_cata)
                            cata_ID.append(i)
    if a2 == 2:
        for i in range(MNC_data_reshaped.shape[0]):
            if (MNC_data_reshaped[i][0]) != 0:
                if (MNC_data_reshaped[i][3]) != 0:                    
                    Free_star = MNC_data_reshaped[i][0]
                    Free_OH = MNC_data_reshaped[i][3]                     
                    single_cata = ((Free_OH+(1/2)*gh2)-(Free_star+gw))*27.11
                    if single_cata < 10:
                        if single_cata > -10:
                            cata_binding.append(single_cata)
                            cata_ID.append(i) 
    if a2 == 3:
        for i in range(MNC_data_reshaped.shape[0]):
            if (MNC_data_reshaped[i][0]) != 0:
                if (MNC_data_reshaped[i][4]) != 0:                    
                    Free_star = MNC_data_reshaped[i][0]
                    Free_H = MNC_data_reshaped[i][4]                     
                    single_cata = ((Free_H)-(Free_star+(1/2)*gh2))*27.11
                    if single_cata < 10:
                        if single_cata > -10:
                            cata_binding.append(single_cata)
                            cata_ID.append(i)
                            
    density_matrices_full = density_matrices_full[cata_ID]
    density_matrices_3d = density_matrices_3d[cata_ID]

    mulliken_orbital_full = mulliken_orbital_full[cata_ID]
    mulliken_orbital_3d = mulliken_orbital_3d[cata_ID]
    mulliken_atomic = mulliken_atomic[cata_ID]

    lowdin_orbital_full = lowdin_orbital_full[cata_ID]
    lowdin_orbital_3d = lowdin_orbital_3d[cata_ID]
    lowdin_atomic = lowdin_atomic[cata_ID]

    cata_binding = np.array(cata_binding).reshape(-1,1)
    
    type_list_1 = np.array(type_list_1)
    type_list_2 = np.array(type_list_2)
    type_list_1 = type_list_1[cata_ID]
    type_list_2 = type_list_2[cata_ID]
                        
    return density_matrices_full, density_matrices_3d, mulliken_orbital_full, mulliken_orbital_3d, mulliken_atomic, lowdin_orbital_full, lowdin_orbital_3d, lowdin_atomic, cata_ID, cata_binding, type_list_1, type_list_2

#%%

orbital_list = [(35, 'Fe', '1s', ''), (35, 'Fe', '2s', ''), (35, 'Fe', '3s', ''), (35, 'Fe', '4s', ''), (35, 'Fe', '2p', 'x'), (35, 'Fe', '2p', 'y'), (35, 'Fe', '2p', 'z'), (35, 'Fe', '3p', 'x'), (35, 'Fe', '3p', 'y'), (35, 'Fe', '3p', 'z'), (35, 'Fe', '4p', 'x'), (35, 'Fe', '4p', 'y'), (35, 'Fe', '4p', 'z'), (35, 'Fe', '3d', 'xy'), (35, 'Fe', '3d', 'yz'), (35, 'Fe', '3d', 'z^2'), (35, 'Fe', '3d', 'xz'), (35, 'Fe', '3d', 'x2-y2'),
                (36, 'N', '1s', ''), (36, 'N', '2s', ''), (36, 'N', '2p', 'x'), (36, 'N', '2p', 'y'), (36, 'N', '2p', 'z'), (37, 'N', '1s', ''), (37, 'N', '2s', ''), (37, 'N', '2p', 'x'), (37, 'N', '2p', 'y'), (37, 'N', '2p', 'z'), (38, 'N', '1s', ''), (38, 'N', '2s', ''), (38, 'N', '2p', 'x'), (38, 'N', '2p', 'y'), (38, 'N', '2p', 'z'), (39, 'N', '1s', ''), (39, 'N', '2s', ''), (39, 'N', '2p', 'x'), (39, 'N', '2p', 'y'), (39, 'N', '2p', 'z')]

q1 = 0
for i in range(len(orbital_list)):
    if orbital_list[i][2] == '3d':
        print(i,orbital_list[i])
        q1 += 1    

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
                    print(orbital_list[i],orbital_list[j])
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
                    print(orbital_list[i],orbital_list[j])
                    q1 += 1
            q2 += 1

matrix_orb_list = np.array(matrix_orb_list)
matrix_3d_list = np.array(matrix_3d_list)

#%%

data_HF_OOH = data(0,0)
data_HF_O = data(0,1)
data_HF_OH = data(0,2)
data_HF_H = data(0,3)

data_DFT_OOH = data(1,0)
data_DFT_O = data(1,1)
data_DFT_OH = data(1,2)
data_DFT_H = data(1,3)

total_data = [[data_HF_OOH,data_HF_O,data_HF_OH,data_HF_H],[data_DFT_OOH,data_DFT_O,data_DFT_OH,data_DFT_H]]

#%%

file_features = '/Users/maximsecor/Desktop/MNC/Cleaned_ANN_input_Data/OOH_UHF/UHF_density_matrices_full.csv'
df_features = pd.read_csv(file_features)
features = df_features.values

pca = PCA(n_components=256)
pca.fit(features)
features_pca = pca.transform(features)

#%%

# print(pca.get_covariance())
# print(np.sum(pca.explained_variance_ratio_))
# print(pca.singular_values_)
plt.scatter(features_pca[:,0],features_pca[:,1])

#%%

total_x = np.array([])
total_y = np.array([])

for i3 in range(10):

    x_list = []
    y_list = []
    data_x = total_data[0][0]
    data_y = total_data[0][0]
    
    for i in range(len(data_x[8])):
        if data_x[8][i] in data_y[8]:
            if data_x[11][i] == i3:
                idy = (np.where(np.array(data_y[8]) == data_x[8][i])[0][0])
                
                x_list.append(RFE[i,6])
                y_list.append(data_y[9][i])
                
                # x_list.append(features[i,0])
                # y_list.append(features[i,2])
    
    if len(total_x) == 0:       
        temp_x = np.array(x_list)
        temp_y = np.array(y_list)              
        total_x = temp_x             
        total_y = temp_y
    else: 
        temp_x = np.array(x_list)
        temp_y = np.array(y_list)           
        total_x = np.concatenate((total_x,temp_x))              
        total_y = np.concatenate((total_y,temp_y))

    plt.scatter(x_list,y_list,c = c_list[i3],edgecolor = e_list[i3],alpha=1,linewidth=0.5,s=12,label=l_list[i3])
    
# total_x = total_x.reshape(-1,1)
# total_y = total_y.reshape(-1,1)    
# reg = LinearRegression().fit(total_x, total_y)
# score = reg.score(total_x, total_y)
# coef = reg.coef_
# inter = reg.intercept_
# print(coef[0],inter[0],score)
# dom = np.linspace(np.min(total_x),np.max(total_x),1024)
# y_val = coef[0]*dom + inter
# plt.plot(dom,y_val,'k', linewidth=2, zorder=-1)

# plt.ylim(-6,4)
# plt.legend(fontsize='x-small')
# plt.savefig("/Users/maximsecor/Desktop/PCA_2.tif",dpi=800)
plt.show()

#%%

start = time.time()

data_y = total_data[0][0][4][:,:5]
data_x = total_data[0][0][4][:,5:]

total = data_x + data_y

df_total = pd.DataFrame(total)
pearson_corr = df_total.corr().values
corr_sort = np.argsort(np.abs(pearson_corr[1:,0]))

end = time.time()

print(end-start)

#%%

print(corr_sort[::-1])
print(np.abs(pearson_corr[1:,0])[corr_sort[::-1]])

#%%

for i in range(5):
    print(corr_sort[::-1][i],matrix_orb_list[corr_sort[::-1][i]],np.abs(pearson_corr[1:,0])[corr_sort[::-1]][i])

#%%

plt.hist(np.abs(pearson_corr[1:,0]))




            
