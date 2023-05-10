
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
print(df.nunique(axis=0))
#np.savetxt('./columnCount.txt', df.nunique(axis=0), fmt='%s')

#另存所有特征
#np.savetxt('./New_Columns.txt', df.columns.values, fmt='%s')

#挑出所有攻擊種類
print(df.conn_state.unique().tolist())
