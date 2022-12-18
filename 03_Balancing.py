'''
03_Repalce.m
1.將多數樣本變少(type=1 減少到與 type=2 同多)
2.採用 undersampling random

------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    增添了創建資料夾 createFolder(path_save + "03_Balancing")
------------------------
'''

import os
import datetime
import numpy as np
import pandas as pd
import re
import ipaddress
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler # 导入欠采样处理库Random UnderSampler


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# load index from 'path' text file
def loadColumns(path):
    file = open(path, "r")
    index = [line.strip() for line in file]
    file.close()
    return index



path_r_raw_mapped_dataset = 'Data/01_Normalization/'
path_r_raw_replace_dataset_Label = 'Data/02_Repalce/'
path_save = 'Data/'

# Create Folder
createFolder(path_save + "03_Balancing")

counter_File = 0

startTime = datetime.datetime.now()
print('==================== Start: ', startTime, '====================')

for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith("_Trn_Big_Normalization.csv"):
        fileName = file.replace("_Normalization.csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # load raw_dataset
        df_data = pd.read_csv(path_r_raw_mapped_dataset + fileName + "_Normalization.csv", header=0, low_memory=False)
        df_label = pd.read_csv(path_r_raw_replace_dataset_Label + fileName + "_TypeLabel_2type.csv", header=0, low_memory=False)
        print(df_data)
        print(df_label)


        UnderSampler = RandomUnderSampler() # 建立RandomUnderSampler模型
        train_data, train_label = UnderSampler.fit_resample(df_data, df_label)  # 输入数据并进行欠采样处理
        # X_over, y_over = undersample.fit_resample(x_train, y_train)

        print('train_label:', train_label)
        print('train_label:', len(train_label))

        type_name = train_label['type']
        # print(type_name)

        count_type_name = pd.Series(list(type_name)).value_counts()
        print(count_type_name)

        # Save to CSV
        train_data.to_csv(path_save + "03_Balancing/" + fileName + "_Normalization_UnderSampling.csv", index=False)
        print("\tSaved: " + path_save + "03_Balancing/" + fileName + "_Normalization_UnderSampling.csv")
        train_label.to_csv(path_save + "03_Balancing/" + fileName + "_TypeLabel_2type_UnderSampling.csv", index=False)
        print("\tSaved: " + path_save + "03_Balancing/" + fileName + "_TypeLabel_2type_UnderSampling.csv")


        for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
            if file.endswith("_Trn_Small_Normalization.csv"):
                fileName = file.replace("_Normalization.csv", '')
                print('\n', counter_File, ':\t', fileName)
                counter_File += 1

                # load raw_dataset
                df_data = pd.read_csv(path_r_raw_mapped_dataset + fileName + "_Normalization.csv", header=0, low_memory=False)
                df_label = pd.read_csv(path_r_raw_replace_dataset_Label + fileName + "_TypeLabel_2type.csv", header=0, low_memory=False)
                print(df_data)
                print(df_label)

                UnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型
                train_data, train_label = UnderSampler.fit_resample(df_data, df_label)  # 输入数据并进行欠采样处理
                # X_over, y_over = undersample.fit_resample(x_train, y_train)

                print('train_label:', train_label)
                print('train_label:', len(train_label))

                type_name = train_label['type']
                # print(type_name)

                count_type_name = pd.Series(list(type_name)).value_counts()
                print(count_type_name)

                # Save to CSV
                train_data.to_csv(path_save + "03_Balancing/" + fileName + "_Normalization_UnderSampling.csv", index=False)
                print("\tSaved: " + path_save + "03_Balancing/" + fileName + "_Normalization_UnderSampling.csv")
                train_label.to_csv(path_save + "03_Balancing/" + fileName + "_TypeLabel_2type_UnderSampling.csv", index=False)
                print("\tSaved: " + path_save + "03_Balancing/" + fileName + "_TypeLabel_2type_UnderSampling.csv")


endTime = datetime.datetime.now()
print('==================== DONE: ', endTime, '====================\n')
print('Duration:', endTime - startTime)