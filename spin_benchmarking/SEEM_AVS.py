#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:44:44 2023

@author: maximsecor
"""

import numpy as np
import os

#%%

loglist = []
temp = os.listdir('/Users/maximsecor/Desktop/ionicBC') 
for i in range(len(temp)):
    if temp[i].split('.')[1] == 'log':
        loglist.append(temp[i])

print(loglist)

for i in range(len(loglist)):
    print(loglist[i])

#%%

for k in range(len(loglist)):

    file = '/Users/maximsecor/Desktop/ionicBC/'+str(loglist[k]) 
    
    a_file = open(file, "r")
    list_of_lists = []
    for line in a_file:
      stripped_line = line.strip()
      line_list = stripped_line.split()
      list_of_lists.append(line_list)
    a_file.close()
    
    list_of_lists = np.array(list_of_lists,dtype=object)
    
    for i in range(len(list_of_lists)):
        temp = list_of_lists[i]
        if len(temp)>0:
            if temp[0] == 'Numb':
                for j in range(5):
                    print(list_of_lists[i+j+2][0],list_of_lists[i+j+2][1],list_of_lists[i+j+2][4])
 
    # for i in range(len(list_of_lists)):
    #     temp = list_of_lists[i]
    #     if len(temp)>0:
    #         if temp[0] == 'Numb':
    #             print(list_of_lists[i-3][2].split('(')[0],list_of_lists[i-3][4].split('(')[0],list_of_lists[i-3][6].split('(')[0])










