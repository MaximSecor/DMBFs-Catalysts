#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:11:23 2023

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
                
    Sc_list = np.linspace(0,203,204,dtype=int)
    Cu_Zn_list = np.linspace(204*8,204*10-1,204*2,dtype=int)
    undoped = np.array([2040,2048,2049,2050,2058,2059])
    exclude_list = np.concatenate((Sc_list,Cu_Zn_list,undoped))
    
    gh2 = -1.180098
    gw = -76.419936
    
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
    
    qq = 0
    for i in range(MNC_data_reshaped.shape[0]):
        if i not in exclude_list:
            if (MNC_data_reshaped[i][4]) != 0:
                qq += 1
    print(qq)
    
    if a2 == 0:
        for i in range(MNC_data_reshaped.shape[0]):
            if i not in exclude_list:
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
            if i not in exclude_list:
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
            if i not in exclude_list:
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
            if i not in exclude_list:
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

print(1048 +1099 +1315 +1236 +1148)

#%%

print(total_data[0][0][11])

#%%

"Figure 6"

c_list = ['lightgreen','bisque','lightblue','lightcoral']
e_list = ['green','orange','blue','red']
l_list = ['Step 1','Step 2','Step 3','Step 4']
l_list_2 = ['Step 4','Step 3','Step 2','Step 1']

data_ooh = total_data[0][0]
data_o = total_data[0][1]
data_oh = total_data[0][2]
data_h = total_data[0][3]

cata_ORR = []
cata_ORR_step = []
cata_OER = []
cata_OER_step = []

ooh_list = []
o_list = []
oh_list = []
h_list = []

g1_list = []
g2_list = []
g3_list = []
g4_list = []

metal_type = []

cata_id = []
q = 0

for i in range(len(data_ooh[8])):
    if data_ooh[8][i] in data_o[8]:
        if data_ooh[8][i] in data_oh[8]:
            if data_ooh[8][i] in data_h[8]:
                    
                q = q + 1 
                
                metal_type.append(data_ooh[11][i])
            
                id_o = np.where(np.array(data_o[8]) == data_ooh[8][i])[0][0]
                id_oh = np.where(np.array(data_oh[8]) == data_ooh[8][i])[0][0]
                id_h = np.where(np.array(data_h[8]) == data_ooh[8][i])[0][0]
                
                g1 = data_ooh[9][i] - 4.92
                g2 = data_o[9][id_o] - data_ooh[9][i]
                g3 = data_oh[9][id_oh] - data_o[9][id_o]
                g4 = -data_oh[9][id_oh]

                g5 = -1*g4
                g6 = -1*g3
                g7 = -1*g2
                g8 = -1*g1
                
                cata_ORR.append(-1*np.max(np.array([g1,g2,g3,g4])))
                cata_ORR_step.append(np.argmax(np.array([g1,g2,g3,g4])))
                cata_OER.append(np.max(np.array([g5,g6,g7,g8])))
                cata_OER_step.append(np.argmax(np.array([g5,g6,g7,g8])))
                
                ooh_list.append(data_ooh[9][i])
                o_list.append(data_o[9][id_o])
                oh_list.append(data_oh[9][id_oh])
                h_list.append(data_h[9][id_h])
                
                g1_list.append(g1)
                g2_list.append(g2)
                g3_list.append(g3)
                g4_list.append(g4)
                
                cata_id.append(data_ooh[8][i])

cata_ORR = np.array(cata_ORR)
cata_ORR_step = np.array(cata_ORR_step)
cata_OER = np.array(cata_OER)
cata_OER_step = np.array(cata_OER_step)

ooh_list = np.array(ooh_list)
o_list = np.array(o_list)
oh_list = np.array(oh_list)
h_list = np.array(h_list)

g1_list = np.array(g1_list)
g2_list = np.array(g2_list)
g3_list = np.array(g3_list)
g4_list = np.array(g4_list)

cata_id = np.array(cata_id)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1

for i in range(4):
    x_data = []
    y_data = []
    for i2 in range(len(cata_ORR_step)):
        if cata_ORR_step[i2] == i:
            x_data.append(oh_list[i2])
            y_data.append(cata_ORR[i2])
    plt.scatter(x_data,y_data,c = c_list[i],edgecolor = e_list[i],alpha=1,linewidth=0.5,s=12,label=l_list[i])
plt.legend(fontsize='x-small') 
plt.savefig("/Users/maximsecor/Desktop/test_ORR.tif",dpi=800)
plt.show()

for i in range(4):
    x_data = []
    y_data = []
    for i2 in range(len(cata_OER_step)):
        if cata_ORR_step[i2] == i:
            x_data.append(oh_list[i2])
            y_data.append(cata_OER[i2])
    plt.scatter(x_data,y_data,c = c_list[::-1][i],edgecolor = e_list[::-1][i],alpha=1,linewidth=0.5,s=12,label=l_list_2[::-1][i])
plt.legend(fontsize='x-small') 
plt.savefig("/Users/maximsecor/Desktop/test_OER.tif",dpi=800)
plt.show()

# plt.scatter(oh_list,cata_OER,c = c_list[i3],edgecolor = e_list[i3],alpha=1,linewidth=0.5,s=12,label=l_list[i3])
# plt.scatter(h_list,-np.abs(h_list),c = c_list[i3],edgecolor = e_list[i3],alpha=1,linewidth=0.5,s=12,label=l_list[i3])

# plt.legend(fontsize='x-small')        
# plt.savefig("/Users/maximsecor/Desktop/test_ORR.tif",dpi=800)
# plt.savefig("/Users/maximsecor/Desktop/test_OER.tif",dpi=800)        
# plt.savefig("/Users/maximsecor/Desktop/test_HER.tif",dpi=800) 

# plt.savefig("/Users/maximsecor/Desktop/test.tif",dpi=800)       
        
# print(np.array(total_oh).reshape(len(total_oh),))

#%%

c_list = ['lightgreen','bisque','lightgreen','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','green','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']

for i in range(7):
    i = i+2
    x_data = []
    y_data = []
    for i2 in range(len(metal_type)):
        if metal_type[i2] == i:
            x_data.append(oh_list[i2])
            y_data.append(cata_OER[i2])
    plt.scatter(x_data,y_data,c = c_list[i],edgecolor = e_list[i],alpha=1,linewidth=0.5,s=12,label=l_list[i])
plt.legend(fontsize='x-small') 
plt.savefig("/Users/maximsecor/Desktop/test.tif",dpi=800)
plt.show()

#%%

print(cata_id)

#%%

[cata_ORR, cata_OER, cata_id]

#%%

print(len(cata_id))

#%%

cata_HER = total_data[0][3][9].reshape(total_data[0][3][9].shape[0])
cata_id_HER = np.array(total_data[0][3][8])

#%%

cata_total = []
list_ORR_HER = []
list_ORR = []
list_HER = []
list_oh = []
for i,j in enumerate(cata_id):
    if j in cata_id_HER:
        cata_total.append(j)
        temp_idx = np.where(cata_id_HER==j)[0][0]
        list_ORR_HER.append(-np.abs(cata_HER[temp_idx])+cata_ORR[i])
        list_ORR.append(cata_ORR[i])
        list_HER.append(-np.abs(cata_HER[temp_idx]))
        list_oh.append(oh_list[i])

cata_total = np.array(cata_total)

# print(np.sort(list_ORR_HER))
list_ORR_HER_sorted = np.sort(list_ORR_HER)
list_ORR_HER_argsorted = np.argsort(list_ORR_HER)

for i in range(5):
    print('\n')
    print(list_ORR_HER_sorted[-i-1])
    print(list_ORR[list_ORR_HER_argsorted[-i-1]])
    print(list_HER[list_ORR_HER_argsorted[-i-1]])
    print(cata_total[list_ORR_HER_argsorted[-i-1]])
    print((cata_total[list_ORR_HER_argsorted[-i-1]]-20)//204)
    print((cata_total[list_ORR_HER_argsorted[-i-1]]-20)%204)
    
#%%

plt.scatter(list_oh,list_ORR_HER)

#%%

test = ['dog','cat','mouse']

for i,k in enumerate(test):
    print(i,k)

#%%

for i in cata_total:
    print(i)
        

 #%%

# print(np.min(cata_ORR))
# print(np.max(cata_ORR))
# print(np.min(cata_OER))
# print(np.max(cata_OER))

# print(cata_id[np.argmin(cata_ORR)])
# print(cata_id[np.argmax(cata_ORR)])
# print(cata_id[np.argmin(cata_OER)])
# print(cata_id[np.argmax(cata_OER)])

print(cata_id[np.argmin(cata_ORR)]//204)
print(cata_id[np.argmax(cata_ORR)]//204)
print(cata_id[np.argmin(cata_OER)]//204)
print(cata_id[np.argmax(cata_OER)]//204)

print(cata_id[np.argmin(cata_ORR)]%204)
print(cata_id[np.argmax(cata_ORR)]%204)
print(cata_id[np.argmin(cata_OER)]%204)
print(cata_id[np.argmax(cata_OER)]%204)

print(810//204)
print(810%204)
print(1203//204)
print(1203%204)

#%%

""" FIGURE 2 """

e_list = ['green','orange','blue','red']

q = np.where(cata_ORR_step==1)[0][0]

print(cata_id[q])
print(cata_ORR_step[q])
print(cata_ORR[q])

s0 = 4.92
s1 = s0 + g1_list[q][0]
s2 = s1 + g2_list[q][0]
s3 = s2 + g3_list[q][0]
s4 = s3 + g4_list[q][0]

# plt.xlim(0,9)
# plt.ylim(-5.5,5.5)

plt.plot((0,1),(s0,s0),'k')
plt.plot((1,2),(s0,s1),e_list[0], linestyle = 'dashed')

plt.plot((2,3),(s1,s1),'k')
plt.plot((3,4),(s1,s2),e_list[1], linestyle = 'dashed')

plt.plot((4,5),(s2,s2),'k')
plt.plot((5,6),(s2,s3),e_list[2], linestyle = 'dashed')

plt.plot((6,7),(s3,s3),'k')
plt.plot((7,8),(s3,s4),e_list[3], linestyle = 'dashed')

plt.plot((8,9),(s4,s4),'k')

plt.show()

#%%

s0 = 4.92
s1 = s0 - s0/4
s2 = s1 - s0/4
s3 = s2 - s0/4
s4 = s3 - s0/4

# plt.xlim(0,9)
# plt.ylim(-5.5,5.5)

plt.plot((0,1),(s0,s0),'k')
plt.plot((1,2),(s0,s1), e_list[0], linestyle = 'dashed')

plt.plot((2,3),(s1,s1),'k')
plt.plot((3,4),(s1,s2),e_list[1], linestyle = 'dashed')

plt.plot((4,5),(s2,s2),'k')
plt.plot((5,6),(s2,s3),e_list[2], linestyle = 'dashed')

plt.plot((6,7),(s3,s3),'k')
plt.plot((7,8),(s3,s4),e_list[3], linestyle = 'dashed')

plt.plot((8,9),(s4,s4),'k')

plt.show()

#%%

"Figure 5"

c_list = ['lightgreen','bisque','lightgreen','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','green','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']

for i1 in range(4):
    for i2 in range(4):
        
        # print(i1,i2)
        
        total_x = np.array([])
        total_y = np.array([])
        
        for i3 in range(10):
            
            if i3 not in [0,8,9]:
            
            # print(i3)

                x_list = []
                y_list = []
                data_x = total_data[0][i1]
                data_y = total_data[0][i2]
                
                for i in range(len(data_x[8])):
                    if data_x[8][i] in data_y[8]:
                        if data_x[11][i] == i3:
                            idy = (np.where(np.array(data_y[8]) == data_x[8][i])[0][0])
                            x_list.append(data_x[9][i])
                            y_list.append(data_y[9][idy])
                
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
            
        reg = LinearRegression().fit(total_x, total_y)
        score = reg.score(total_x, total_y)
        coef = reg.coef_
        inter = reg.intercept_
        print(i1,i2,coef[0],inter[0],score)
        dom = np.linspace(np.min(total_x),np.max(total_x),1024)
        y_val = coef[0]*dom + inter
        plt.plot(dom,y_val,'k', linewidth=2, zorder=-1)
        
        plt.legend(fontsize='x-small')
        plt.savefig("/Users/maximsecor/Desktop/bindings_metal_"+str(i1)+str(i2)+".tif",dpi=800)
        plt.show()

#%%

c_list = ['lightgreen','bisque','cyan','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','darkcyan','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Bare','1D','2D','3D','H2O','H2O 1D','H2O 2D','H2O 3D']

for i1 in range(4):
    for i2 in range(4):

        total_x = np.array([])
        total_y = np.array([])
        
        for i3 in range(8):

            x_list = []
            y_list = []
            
            data_x = total_data[0][i1]
            data_y = total_data[0][i2]
            
            for i in range(len(data_x[8])):
                if data_x[8][i] in data_y[8]:
                    if data_x[10][i] == i3:
                        idy = (np.where(np.array(data_y[8]) == data_x[8][i])[0][0])
                        x_list.append(data_x[9][i])
                        y_list.append(data_y[9][idy])
            
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
            
        reg = LinearRegression().fit(total_x, total_y)
        score = reg.score(total_x, total_y)
        coef = reg.coef_
        inter = reg.intercept_

        dom = np.linspace(np.min(total_x),np.max(total_x),1024)
        y_val = coef[0]*dom + inter
        plt.plot(dom,y_val,'k', linewidth=2, zorder=-1)
        
        plt.legend(fontsize='x-small')
        # plt.savefig("/Users/maximsecor/Desktop/bindings_dopant_"+str(i1)+str(i2)+".tif",dpi=800)
        plt.show()
                
#%%

refs = (data_x[9][-20:-10])
refs_1 = (data_x[9][-10:])

print(refs,refs_1)

#%%

c_list = ['lightgreen','bisque','cyan','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','darkcyan','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Bare','1D','2D','3D','H2O','H2O 1D','H2O 2D','H2O 3D']

for i1 in range(4):
    for i2 in range(4):

        total_x = np.array([])
        total_y = np.array([])
        
        for i3 in range(8):

            x_list = []
            y_list = []
            
            data_x = total_data[0][i1]
            data_y = total_data[0][i2]
            
            ref_1 = data_x[9][-20:-10]
            ref_2 = data_y[9][-20:-10]
            
            for i in range(len(data_x[8])):
                if data_x[8][i] in data_y[8]:
                    if data_x[10][i] == i3:
                        idy = (np.where(np.array(data_y[8]) == data_x[8][i])[0][0])
                        x_list.append(data_x[9][i]-ref_1[data_x[11][i]])
                        y_list.append(data_y[9][idy]-ref_2[data_y[11][idy]])
            
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
            
        reg = LinearRegression().fit(total_x, total_y)
        score = reg.score(total_x, total_y)
        coef = reg.coef_
        inter = reg.intercept_

        dom = np.linspace(np.min(total_x),np.max(total_x),1024)
        y_val = coef[0]*dom + inter
        plt.plot(dom,y_val,'k', linewidth=2, zorder=-1)
        
        plt.legend(fontsize='x-small')
        # plt.savefig("/Users/maximsecor/Desktop/bindings_dopant_refed"+str(i1)+str(i2)+".tif",dpi=800)
        plt.show()

#%%

print(data_x[11])

#%%

for q in range(4):
    c_list = ['lightcoral','lightblue']
    e_list = ['red','blue','darkcyan']
    
    for i3 in range(2):
        x_list = []
        data_x = total_data[0][q]
        for i in range(len(data_x[8])):
            if data_x[10][i] in np.array([0,1,2,3])+i3*np.array([4,4,4,4]):
                x_list.append(data_x[9][i])
        x_list = np.array(x_list)
        bin_list = np.linspace(np.min(data_x[9]),np.max(data_x[9]),20)
        sns.histplot(x_list.reshape(len(x_list)),color=c_list[i3],bins=bin_list,alpha=0.5)
    
    plt.savefig("/Users/maximsecor/Desktop/test_"+str(q)+".tif",dpi=800)
    plt.show()

#%%

print(data_x[9])

#%%

temp = [1,2,3,4]
temp = np.array(temp)

print(np.where(temp==2))

#%%

for q in range(4):
    for i in range(10):
        if i not in [0,8,9]:
            x_list = []
            data_x = total_data[0][q]
            for j in range(len(data_x[8])):
                if data_x[11][j] == i:
                    if (data_x[8][j] + 102) in data_x[8]: 
                        k = np.where(np.array(data_x[8])==data_x[8][j]+102)[0][0]
                        if data_x[11][j] == data_x[11][k]:
                            x_list.append((data_x[9][k]-data_x[9][j])[0])
    
    x_list = np.array(x_list)
    print(len(x_list))
    bin_list = np.linspace(0,2,20)
    plt.xlim(0,2)
    sns.histplot(x_list.reshape(len(x_list)),color=c_list[i3],bins=bin_list,alpha=0.5)
    plt.show()
                        

#%%

dopant_ID = []
for i in range(10):
    for j in  range(102):
        dopant_ID.append(j)
    for j in  range(102):
        dopant_ID.append(j)   
for i in range(20):
    dopant_ID.append(0)
dopant_ID = np.array(dopant_ID)
print(dopant_ID.shape)

#%%

x_list = []
data_x = total_data[0][3]

code_list = []

for i in range(len(data_x[8])):
    code_list.append(str(data_x[10][i])+'_'+str(data_x[11][i])+'_'+str(dopant_ID[data_x[8][i]]))

for i in range(len(data_x[8])):
    if data_x[10][i] in [0,1,2,3]:

        temp_code_1 = (str(data_x[10][i])+'_'+str(data_x[11][i])+'_'+str(dopant_ID[data_x[8][i]]))        
        temp_code_2 = (str(data_x[10][i]+4)+'_'+str(data_x[11][i])+'_'+str(dopant_ID[data_x[8][i]]))
        
        if temp_code_2 in code_list:
            
            ID_nowater = np.where(np.array(code_list) == temp_code_1)[0][0]
            ID_water = np.where(np.array(code_list) == temp_code_2)[0][0]
            
            x_list.append(data_x[9][ID_water]-data_x[9][ID_nowater])

    
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2.5

x_list = np.array(x_list)
bin_list = np.linspace(np.min(x_list),np.max(x_list),20)
sns.histplot(x_list.reshape(len(x_list)),color=c_list[i3],bins=bin_list,alpha=0.5,linewidth=2.5)

plt.xlabel("")
plt.ylabel("")

# plt.savefig("/Users/maximsecor/Desktop/dopants_effect_"+str(i1)+".tif",dpi=800)
plt.show()


#%%

for i1 in range(4):

    x_list = []
    hue_list = []
    data_x = total_data[0][i1]
    
    for i2 in range(3):
        for i in range(20):
            
            dope_ID = data_x[10][-20:][i]
            metal_ID = data_x[11][-20:][i]
            ref_binding = data_x[9][-20:][i]
            
            # print(dope_ID,metal_ID)
            
            for j in range(len(data_x[8])):
                if data_x[10][j] == dope_ID + 1 + i2:
                    if data_x[11][j] == metal_ID:
                        x_list.append((data_x[9][j]-ref_binding)[0])
                        hue_list.append(i2)
    
    mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 1
    
    x_list = np.array(x_list)
    hue_list = np.array(hue_list)
    
    plot_list = np.concatenate((x_list.reshape(-1,1),hue_list.reshape(-1,1)),1)
    plot_list_df = pd.DataFrame(plot_list, columns = ['x_list','hue_list'])
    
    c_list = ['lightgreen','lightcoral','plum']
    
    bin_list = np.linspace(np.min(x_list),np.max(x_list),20)
    sns.histplot(data=plot_list_df, x="x_list", hue="hue_list", multiple="stack",bins=bin_list,alpha=0.5,legend=False,palette=c_list,linewidth=1)
    
    plt.xlabel("")
    plt.ylabel("")
    
    plt.xlim(-0.5,1.5)
    
    # plt.savefig("/Users/maximsecor/Desktop/dopants_effect_"+str(i1)+".tif",dpi=800)
    plt.show()
            
        
#%%

"""HER"""

c_list = ['lightgreen','bisque','cyan','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','darkcyan','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Bare','1D','2D','3D','H2O','H2O 1D','H2O 2D','H2O 3D']



i1 = 3
i2 = 3

total_x = np.array([])
total_y = np.array([])

for i3 in range(8):

    x_list = []
    y_list = []
    
    data_x = total_data[0][i1]
    data_y = total_data[0][i2]
    
    ref_1 = data_x[9][-20:-10]
    ref_2 = data_y[9][-20:-10]
    
    for i in range(len(data_x[8])):
        if data_x[8][i] in data_y[8]:
            if data_x[10][i] == i3:
                idy = (np.where(np.array(data_y[8]) == data_x[8][i])[0][0])
                x_list.append(data_x[9][i]-ref_1[data_x[11][i]])
                y_list.append(data_y[9][idy]-ref_2[data_y[11][idy]])
    
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
    
reg = LinearRegression().fit(total_x, total_y)
score = reg.score(total_x, total_y)
coef = reg.coef_
inter = reg.intercept_

dom = np.linspace(np.min(total_x),np.max(total_x),1024)
y_val = coef[0]*dom + inter
plt.plot(dom,y_val,'k', linewidth=2, zorder=-1)

plt.legend(fontsize='x-small')
# plt.savefig("/Users/maximsecor/Desktop/HER_1.tif",dpi=800)
plt.show()

#%%

c_list = ['lightgreen','bisque','lightgreen','lightcoral','plum','wheat','lightpink','lightblue','honeydew','gainsboro']
e_list = ['green','orange','green','red','purple','brown','deeppink','blue','olive','gray']
l_list = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']

min_val = 10
place = 0

for i3 in range(10):
    if i3 not in [0,8,9]:
        x_list = []
        data_x = total_data[0][3]
        for i in range(len(data_x[8])):
                if data_x[11][i] == i3:
                    x_list.append(data_x[9][i])
        
                    if abs(data_x[9][i]) < min_val:
                        min_val = abs(data_x[9][i])
                        print(data_x[8][i],data_x[9][i],data_x[10][i],data_x[11][i])
                    
                    # if abs(data_x[9][i]) > min_val:
                    #     min_val = abs(data_x[9][i])
                    #     print(data_x[8][i],data_x[9][i],data_x[10][i],data_x[11][i])
                        
                        
    
        plt.scatter(x_list,-np.abs(x_list),c = c_list[i3],edgecolor = e_list[i3],alpha=1,linewidth=0.5,s=12,label=l_list[i3])
    
plt.legend(fontsize='x-small')
plt.savefig("/Users/maximsecor/Desktop/HER_2.tif",dpi=800)
plt.show()


            
