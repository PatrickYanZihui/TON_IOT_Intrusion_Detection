'''
02_Repalce.m
1.轉換 Type
    flooding  : 3 --> 2
    injection : 4 --> 2
    
------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    增添了創建資料夾的函數createFolder
    增添了創建資料夾 createFolder(path_save + "02_Repalce")
------------------------
'''

import os
import datetime
import numpy as np
import pandas as pd
from collections import Counter

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


# PATH
path_r_raw_mapped_dataset = 'Data/01_Normalization/'
path_save = 'Data/'

# Create Folder
createFolder(path_save + "02_Repalce")

counter_File = 0

startTime = datetime.datetime.now()
print('==================== Start: ', startTime, '====================')

for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith("_TypeLabel.csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # load raw_dataset
        df_data = pd.read_csv(path_r_raw_mapped_dataset + file, sep=',', low_memory=False, index_col=False)
        print(df_data)

        # Type name to Num
        df_data['type'] = df_data['type'].replace(2, 1)
        df_data['type'] = df_data['type'].replace(3, 1)
        df_data['type'] = df_data['type'].replace(4, 1)
        df_data['type'] = df_data['type'].replace(5, 1)
        df_data['type'] = df_data['type'].replace(6, 1)
        df_data['type'] = df_data['type'].replace(7, 1)
        df_data['type'] = df_data['type'].replace(8, 1)
        df_data['type'] = df_data['type'].replace(9, 1)
        df_data['type'] = df_data['type'].replace(10, 1)
        # print(df_data['type'])

        type_name = df_data['type']
        # print(type_name)

        count_type_name = pd.Series(list(type_name)).value_counts()
        print(count_type_name)

        # Save to CSV
        df_data.to_csv(path_save + "02_Repalce/" + fileName + "_2type.csv", index=False)
        print("\tSaved: " + path_save + "02_Repalce/" + fileName + "_2type.csv")

endTime = datetime.datetime.now()
print('==================== DONE: ', endTime, '====================\n')
print('Duration:', endTime - startTime)


# 算一特徵各欄位(字串)出現幾次
for i, file in enumerate(os.listdir(path_save + '02_Repalce/')):
    if file.endswith(".csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        df_data_mapped_2type = pd.read_csv(path_save + '02_Repalce/' + file, sep=',', low_memory=False, index_col=False)
        print(df_data_mapped_2type)

        type_name = df_data_mapped_2type['type']
        print(type_name)

        count_type_name = pd.Series(list(type_name)).value_counts()
        print(count_type_name)

