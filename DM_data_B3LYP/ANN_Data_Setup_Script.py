#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:32:00 2022

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
import os
from os.path import exists
import seaborn as sns
import sys

np.set_printoptions(precision=4,suppress=True)

#%%

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/FreeEnergy.csv'
MNC_data = pd.read_csv(file_MNC)
MNC_data = MNC_data.values

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/FreeEnergy_NoDopant.csv'
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

#%%

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/DensityMatrices.csv'
density_matrices_data = pd.read_csv(file_MNC)
density_matrices_data = density_matrices_data.values

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/DensityMatrices.csv'
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

#%%

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/charge_Fe_list.csv'
temp_data_1 = pd.read_csv(file_MNC)
temp_data_1 = temp_data_1.values
temp_data_1 = temp_data_1.reshape(2040, 2, 3)
file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/charge_Fe_list.csv'
temp_data_2 = pd.read_csv(file_MNC)
temp_data_2 = temp_data_2.values
temp_data_2 = temp_data_2.reshape(20, 2, 3)
charge_Fe_list = np.concatenate((temp_data_1,temp_data_2))

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/charge_N4_list.csv'
temp_data_1 = pd.read_csv(file_MNC)
temp_data_1 = temp_data_1.values
temp_data_1 = temp_data_1.reshape(2040, 8, 3)
file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/charge_N4_list.csv'
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

#%%

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/pop_Fe_list.csv'
temp_data_1 = pd.read_csv(file_MNC)
temp_data_1 = temp_data_1.values
temp_data_1 = temp_data_1.reshape(2040, 36, 2)

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/pop_Fe_list.csv'
temp_data_2 = pd.read_csv(file_MNC)
temp_data_2 = temp_data_2.values
temp_data_2 = temp_data_2.reshape(20, 36, 2)

pop_Fe_list = np.concatenate((temp_data_1,temp_data_2))

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_DOPED/pop_N4_list.csv'
temp_data_1 = pd.read_csv(file_MNC)
temp_data_1 = temp_data_1.values
temp_data_1 = temp_data_1.reshape(2040, 40, 2)

file_MNC = '/Users/maximsecor/Desktop/MNC/Catalytic_Density_Data/UHF_METAL/pop_N4_list.csv'
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

#%%

mulliken_orbital_3d = np.zeros((2060,10))
lowdin_orbital_3d = np.zeros((2060,10))

q1 = 0
for i in range(len(orbital_list)):
    if orbital_list[i][2] == '3d':
        print(i,orbital_list[i])
        for j in range(len(mulliken_orbital_full)):
            mulliken_orbital_3d[j,q1] =  mulliken_orbital_full[j,q1]
            mulliken_orbital_3d[j,q1+5] =  mulliken_orbital_full[j,q1+5]
            lowdin_orbital_3d[j,q1] =  lowdin_orbital_full[j,q1]
            lowdin_orbital_3d[j,q1+5] =  lowdin_orbital_full[j,q1+5]
        q1 += 1    

#%%

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
                    print(orbital_list[i],orbital_list[j])
                    for i2 in range(2060):
                        density_matrices_3d[i2,q1] = density_matrices_full[i2,q2]
                        density_matrices_3d[i2,q1+15] = density_matrices_full[i2,q2+741]
                    q1 += 1
            q2 += 1

#%%

# Create list of excluded Sc, Cu, Zn

Sc_list = np.linspace(0,203,204,dtype=int)
Cu_Zn_list = np.linspace(204*8,204*10-1,204*2,dtype=int)
undoped = np.array([2040,2048,2049,2050,2058,2059])
exclude_list = np.concatenate((Sc_list,Cu_Zn_list,undoped))

#%%

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

#%%

cata_binding = []
cata_ID = []
for i in range(MNC_data_reshaped.shape[0]):
    if i not in exclude_list:
        if (MNC_data_reshaped[i][0]) != 0:
            if (MNC_data_reshaped[i][4]) != 0:
                
                Free_star = MNC_data_reshaped[i][0]
                Free_OOH = MNC_data_reshaped[i][1]
                Free_O = MNC_data_reshaped[i][2]                    
                Free_OH = MNC_data_reshaped[i][3]
                Free_H = MNC_data_reshaped[i][4]
                
                # single_cata = ((Free_OOH+(3/2)*gh2)-(Free_star+2*gw))*27.11
                # single_cata = ((Free_O+gh2)-(Free_star+gw))*27.11
                # single_cata = ((Free_OH+(1/2)*gh2)-(Free_star+gw))*27.11
                single_cata = ((Free_H)-(Free_star+(1/2)*gh2))*27.11
                
                if single_cata < 10:
                    if single_cata > -10:
                        cata_binding.append(single_cata)
                        cata_ID.append(i)
                    else:
                        print(single_cata,"under")
                else:
                    print(single_cata,"over")            

#%%

print(len(cata_ID))

#%%

density_matrices_full = density_matrices_full[cata_ID]
density_matrices_3d = density_matrices_3d[cata_ID]

mulliken_orbital_full = mulliken_orbital_full[cata_ID]
mulliken_orbital_3d = mulliken_orbital_3d[cata_ID]
mulliken_atomic = mulliken_atomic[cata_ID]

lowdin_orbital_full = lowdin_orbital_full[cata_ID]
lowdin_orbital_3d = lowdin_orbital_3d[cata_ID]
lowdin_atomic = lowdin_atomic[cata_ID]

cata_binding = np.array(cata_binding).reshape(-1,1)

#%%

file_density_matrices_full = '/Users/maximsecor/Desktop/MNC_NEW/UHF_density_matrices_full.csv'
file_density_matrices_3d= '/Users/maximsecor/Desktop/MNC_NEW/UHF_density_matrices_3d.csv'

file_mulliken_orbital_full = '/Users/maximsecor/Desktop/MNC_NEW/UHF_mulliken_orbital_full.csv'
file_mulliken_orbital_3d = '/Users/maximsecor/Desktop/MNC_NEW/UHF_mulliken_orbital_3d.csv'
file_mulliken_atomic = '/Users/maximsecor/Desktop/MNC_NEW/UHF_mulliken_atomic.csv'

file_lowdin_orbital_full = '/Users/maximsecor/Desktop/MNC_NEW/UHF_lowdin_orbital_full.csv'
file_lowdin_orbital_3d = '/Users/maximsecor/Desktop/MNC_NEW/UHF_lowdin_orbital_3d.csv'
file_lowdin_atomic = '/Users/maximsecor/Desktop/MNC_NEW/UHF_lowdin_atomic.csv'

file_cata_binding = '/Users/maximsecor/Desktop/MNC_NEW/cata_binding.csv'

#%%

os.system('touch ' + file_density_matrices_full)
os.system('touch ' + file_density_matrices_3d)

os.system('touch ' + file_mulliken_orbital_full)
os.system('touch ' + file_mulliken_orbital_3d)
os.system('touch ' + file_mulliken_atomic)

os.system('touch ' + file_lowdin_orbital_full)
os.system('touch ' + file_lowdin_orbital_3d)
os.system('touch ' + file_lowdin_atomic)

os.system('touch ' + file_cata_binding)

#%%

df_density_matrices_full = pd.DataFrame(density_matrices_full)
df_density_matrices_3d = pd.DataFrame(density_matrices_3d)

df_mulliken_orbital_full = pd.DataFrame(mulliken_orbital_full)
df_mulliken_orbital_3d = pd.DataFrame(mulliken_orbital_3d)
df_mulliken_atomic = pd.DataFrame(mulliken_atomic)

df_lowdin_orbital_full = pd.DataFrame(lowdin_orbital_full)
df_lowdin_orbital_3d = pd.DataFrame(lowdin_orbital_3d)
df_lowdin_atomic = pd.DataFrame(lowdin_atomic)

df_cata_binding = pd.DataFrame(cata_binding)

#%%

df_density_matrices_full.to_csv(file_density_matrices_full, index = False, header=True)
df_density_matrices_3d.to_csv(file_density_matrices_3d, index = False, header=True)

df_mulliken_orbital_full.to_csv(file_mulliken_orbital_full, index = False, header=True)
df_mulliken_orbital_3d.to_csv(file_mulliken_orbital_3d, index = False, header=True)
df_mulliken_atomic.to_csv(file_mulliken_atomic, index = False, header=True)

df_lowdin_orbital_full.to_csv(file_lowdin_orbital_full, index = False, header=True)
df_lowdin_orbital_3d.to_csv(file_lowdin_orbital_3d, index = False, header=True)
df_lowdin_atomic.to_csv(file_lowdin_atomic, index = False, header=True)

df_cata_binding.to_csv(file_cata_binding, index = False, header=True)




