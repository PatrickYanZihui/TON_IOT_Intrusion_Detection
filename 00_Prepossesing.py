'''
------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/12/19 02:49
Description ： 
    First time created the file
    Shuffle and spilt the dataset to train and test
------------------------
'''
import os
import datetime
import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import re
import ipaddress

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

# Path
path_r_Label = 'TON_IOT_Label.txt'
path_r_raw_dataset = 'rawDataSet/'
path_s_data = 'Data/0_DataSet/'

# Create Folder
createFolder(path_s_data)

df = pd.read_csv('./rawDataSet/Train_Test_Network.csv',header=0)

# Count unique values in each row
#print(df.nunique(axis=0))
#np.savetxt('./columnCount.txt', df.nunique(axis=0), fmt='%s')

#另存所有特征
#np.savetxt('./New_Columns.txt', df.columns.values, fmt='%s')

#挑出所有攻擊種類
print(df.type.unique())
np.savetxt('./type.txt', df.type.unique(), fmt='%s')

#前處理
#Shuffle dataset

shuffled_df = df.sample(frac=1)
shuffled_df.to_csv('./rawDataSet/Train_Test_Network_shuffled.csv', index=False)

#Split to train & test data

train_data, test_data = train_test_split(shuffled_df, random_state=777, train_size=0.7)
#print(train_data.shape[0]) #322730
#print(test_data.shape[0]) #138313


# save as csv file
train_data.to_csv('./Data/0_DataSet/Train_Test_Network_Trn_Big.csv', index=False)
test_data.to_csv('./Data/0_DataSet/Train_Test_Network_Tst_Big.csv', index=False)
print("\tSaved: ./Data/0_DataSet/Train_Test_Network_Trn_Big.csv")
print("\tSaved: ./Data/0_DataSet/Train_Test_Network_Tst_Big.csv")