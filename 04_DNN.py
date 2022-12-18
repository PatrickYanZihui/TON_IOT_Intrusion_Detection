'''
1. DNN

------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    注釋了312行的#apply_DNN_SmallData，目前沒有smalldata
------------------------
'''

import os
import datetime
import keras
import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
from collections import Counter
from keras.utils import np_utils
# from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
# from tensorflow.python.keras.utils import plot_model
from matplotlib import pyplot as plt, collections
from pycm import ConfusionMatrix
# from pycm import *
from keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


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

# build DNN model
def build_DNN(input_dim):
    input = Input(shape=(input_dim,))
    x = Dense(input_dim, activation='relu')(input)
    x = Dense(12, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(2, activation='sigmoid')(x)

    # build DNN model
    # 以compile函數定義 優化函數(optimizer)、損失函數(loss)、成效衡量指標(mertrics)
    # loss = 'categorical_crossentropy' 可以在類別超過 2 時使用
    dnn = Model(inputs=input, outputs=output)
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dnn.summary()
    return dnn

def apply_DNN_BigData(normalizeType, num_classes, num_dataset, epoch, batch_size):
    modelType = "DNN"

    ''' PATH '''
    path_r_raw_Normalized_Blanced_TrnDataset = 'Data/03_Balancing/'
    path_r_raw_Normalized_TstData = 'Data/01_Normalization/'
    path_r_raw_2type_TstLabel = 'Data/02_Repalce/'

    path_save = 'Data/'
    path_save_model_big = 'Data/04_DNN/' + "Big_" + normalizeType + "_" + modelType + "/"
    path_r_index = "New_Columns.txt"
    createFolder(path_save_model_big)

    column_list = loadColumns(path_r_index)
    input_dim = len(column_list)

    # build model
    DNN = build_DNN(input_dim)

    # save the best model
    keepBestModel = ModelCheckpoint(filepath=path_save_model_big + "Big_" + normalizeType + "_" + modelType + '_best_model.h5', verbose=1, save_best_only=True)

    startTime = datetime.datetime.now()
    print('==================== START: Big Data', startTime, '====================\n')
    print("Train DNN")
    for e in range(epoch):
        filesName_Set = []
        counter_File = 0

        for f, file in enumerate(os.listdir(path_r_raw_Normalized_Blanced_TrnDataset)):
            if file.endswith('_Trn_Big_Normalization_UnderSampling.csv'):

                fileName = file.replace("_Normalization_UnderSampling.csv", "")
                counter_File += 1

                print("", "Epoch", e, "-", counter_File, ":\t", fileName)
                filesName_Set.append(str(fileName))

                # Load raw_dataset
                train_data = pd.read_csv(path_r_raw_Normalized_Blanced_TrnDataset + fileName + "_Normalization_UnderSampling.csv", header=0, low_memory=False)
                train_label = pd.read_csv(path_r_raw_Normalized_Blanced_TrnDataset + fileName + "_TypeLabel_2type_UnderSampling.csv", header=0, low_memory=False)

                # convert to one hot label
                train_label_oneHot = np_utils.to_categorical(train_label, num_classes=num_classes)

                # train DNN
                # 訓練：以compile函數進行訓練，指定訓練的樣本資料(x(train_data), y(train_label_oneHot))，並撥一部分資料作驗證(validation_split=0.2，將兩成資料作為驗證集)，還有要訓練幾個週期、訓練資料的抽樣方式
                # 進行訓練, 訓練過程會存在 train_history 變數中
                if counter_File == num_dataset:
                    train_history = DNN.fit(train_data, train_label_oneHot, validation_split=0.2, epochs=1, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[keepBestModel]).history
                else :
                    DNN.fit(train_data, train_label_oneHot, validation_split=0.2, epochs=1, batch_size=batch_size, shuffle=True,  verbose=1)
                # DNN.fit(train_data, train_label_oneHot, epochs=epoch, batch_size=batch_size, shuffle=True, validation_split=0.2, verbose=1, callbacks=[keepBestModel])

    # save model structure figure
    plot_model(DNN, to_file=path_save_model_big + "Big_" + modelType + '_structure.png', show_shapes=True)
    # tf.keras.utils.plot_model(DNN, to_file=path_save_model + modelType + '_structure.png', show_shapes=True)

    endTime = datetime.datetime.now()
    print('==================== DONE: BigData', endTime, '====================\n')
    print('Duration:', endTime - startTime)


    '''
    Perform test on the best DNN model
    '''
    print("\n")
    print("\t\t=============================")
    print("\t\t|| Test the best DNN model || Big Data")
    print("\t\t==============================")

    # load the best model
    best_model = load_model(path_save_model_big + "Big_" + normalizeType + "_" + modelType + '_best_model.h5')

    combine_best_model_cm = []
    filesName_Set = []
    print("Test with DNN Big Data")
    for i, file in enumerate(os.listdir(path_r_raw_Normalized_TstData)):
        if file.endswith('_Tst_Big_Normalization.csv'):

            fileName = file.replace('_Normalization.csv', '')
            print('\n', counter_File, ':\t', fileName)
            filesName_Set.append(str(fileName))
            counter_File = counter_File + 1

            # Load raw_dataset
            test_data = pd.read_csv(path_r_raw_Normalized_TstData + fileName + "_Normalization.csv", header=0, low_memory=False)
            test_label = pd.read_csv(path_r_raw_2type_TstLabel + fileName + "_TypeLabel_2type.csv", header=0, low_memory=False)

            test_label_oneHot = np_utils.to_categorical(test_label, num_classes=num_classes)
            test_label = np.argmax(test_label_oneHot, axis=1)

            # perform prediction on test data with the best model
            best_model_predicted_oneHot = best_model.predict(test_data, verbose=0)
            best_model_prediction = np.argmax(best_model_predicted_oneHot, axis=1)

            # create confusion matrix to check accuracy
            combine_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)

            # if i == 0:
            #     # create confusion matrix to check accuracy
            #     combine_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)
            # else:
            #     temp_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)
            #     combine_best_model_cm = combine_best_model_cm.combine(temp_best_model_cm)


            print(combine_best_model_cm.print_matrix())
            combine_best_model_cm.save_html(path_save_model_big + "Big_" +normalizeType + "_DNN_cm2")
            combine_best_model_cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
            plt.title("Big_" + normalizeType + "_DNN")
            plt.savefig(path_save_model_big + "Big_" + normalizeType + "_DNN_best_model.png")

    print("Total files found:\t", counter_File)

    endTime = datetime.datetime.now()
    print('==================== DONE: Big Data', endTime, '====================\n')
    print('Duration:', endTime - startTime, '\n\n\n\n')

def apply_DNN_SmallData(normalizeType, num_classes, num_dataset, epoch, batch_size):
    modelType = "DNN"

    ''' PATH '''
    path_r_raw_Normalized_Blanced_TrnDataset = 'Data/03_Balancing/'
    path_r_raw_Normalized_TstData = 'Data/01_Normalization/'
    path_r_raw_2type_TstLabel = 'Data/02_Repalce/'

    path_save = 'Data/'
    path_save_model_small = 'Data/04_DNN/' + "Small_" + normalizeType + "_" + modelType + "/"
    path_r_index = "New_Columns.txt"
    createFolder(path_save_model_small)

    column_list = loadColumns(path_r_index)
    input_dim = len(column_list)

    # build model
    DNN = build_DNN(input_dim)

    # save the best model
    keepBestModel = ModelCheckpoint(filepath=path_save_model_small + "Small_" + normalizeType + "_" + modelType + '_best_model.h5', verbose=1, save_best_only=True)

    startTime = datetime.datetime.now()
    print('==================== START: Small Data ', startTime, '====================\n')
    print("Train DNN")
    for e in range(epoch):
        filesName_Set = []
        counter_File = 0

        for f, file in enumerate(os.listdir(path_r_raw_Normalized_Blanced_TrnDataset)):
            if file.endswith('_Trn_Small_Normalization_UnderSampling.csv'):

                fileName = file.replace("_Normalization_UnderSampling.csv", "")
                counter_File += 1

                print("", "Epoch", e, "-", counter_File, ":\t", fileName)
                filesName_Set.append(str(fileName))

                # Load raw_dataset
                train_data = pd.read_csv(path_r_raw_Normalized_Blanced_TrnDataset + fileName + "_Normalization_UnderSampling.csv", header=0, low_memory=False)
                train_label = pd.read_csv(path_r_raw_Normalized_Blanced_TrnDataset + fileName + "_TypeLabel_2type_UnderSampling.csv", header=0, low_memory=False)

                # convert to one hot label
                train_label_oneHot = np_utils.to_categorical(train_label, num_classes=num_classes)

                # train DNN
                # 訓練：以compile函數進行訓練，指定訓練的樣本資料(x(train_data), y(train_label_oneHot))，並撥一部分資料作驗證(validation_split=0.2，將兩成資料作為驗證集)，還有要訓練幾個週期、訓練資料的抽樣方式
                # 進行訓練, 訓練過程會存在 train_history 變數中
                if counter_File == num_dataset:
                    train_history = DNN.fit(train_data, train_label_oneHot, validation_split=0.2, epochs=1, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[keepBestModel]).history
                else :
                    DNN.fit(train_data, train_label_oneHot, validation_split=0.2, epochs=1, batch_size=batch_size, shuffle=True,  verbose=1)
                #　DNN.fit(train_data, train_label_oneHot, epochs=epoch, batch_size=batch_size, shuffle=True, validation_split=0.2, verbose=1, callbacks=[keepBestModel])

    # save model structure figure
    plot_model(DNN, to_file=path_save_model_small + "Small_" + modelType + '_structure.png', show_shapes=True)
    # tf.keras.utils.plot_model(DNN, to_file=path_save_model + modelType + '_structure.png', show_shapes=True)

    endTime = datetime.datetime.now()
    print('==================== DONE: Small Data ', endTime, '====================\n')
    print('Duration:', endTime - startTime)


    '''
    Perform test on the best DNN model
    '''
    print("\n")
    print("\t\t=============================")
    print("\t\t|| Test the best DNN model || Small Data")
    print("\t\t==============================")

    # load the best model
    best_model = load_model(path_save_model_small + "Small_" + normalizeType + "_" + modelType + '_best_model.h5')

    combine_best_model_cm = []
    filesName_Set = []
    print("Test with DNN Small Data")
    for i, file in enumerate(os.listdir(path_r_raw_Normalized_TstData)):
        if file.endswith('_Tst_Small_Normalization.csv'):

            fileName = file.replace('_Normalization.csv', '')
            print('\n', counter_File, ':\t', fileName)
            filesName_Set.append(str(fileName))
            counter_File = counter_File + 1

            # Load raw_dataset
            test_data = pd.read_csv(path_r_raw_Normalized_TstData + fileName + "_Normalization.csv", header=0, low_memory=False)
            test_label = pd.read_csv(path_r_raw_2type_TstLabel + fileName + "_TypeLabel_2type.csv", header=0, low_memory=False)

            test_label_oneHot = np_utils.to_categorical(test_label, num_classes=num_classes)
            test_label = np.argmax(test_label_oneHot, axis=1)

            # perform prediction on test data with the best model
            best_model_predicted_oneHot = best_model.predict(test_data, verbose=0)
            best_model_prediction = np.argmax(best_model_predicted_oneHot, axis=1)

            # create confusion matrix to check accuracy
            combine_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)

            # if i == 0:
            #     # create confusion matrix to check accuracy
            #     combine_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)
            # else:
            #     temp_best_model_cm = ConfusionMatrix(test_label, best_model_prediction)
            #     combine_best_model_cm = combine_best_model_cm.combine(temp_best_model_cm)


            print(combine_best_model_cm.print_matrix())
            combine_best_model_cm.save_html(path_save_model_small + "Small_" + normalizeType + "_DNN_cm2")
            combine_best_model_cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
            plt.title("Small_" +normalizeType + "_DNN")
            plt.savefig(path_save_model_small + "Small_" + normalizeType + "_DNN_best_model.png")

    print("Total files found:\t", counter_File)

    endTime = datetime.datetime.now()
    print('==================== DONE: Small Data Test Best Model', endTime, '====================\n')
    print('Duration:', endTime - startTime, '\n\n\n\n')




# Run DNN
num_dataset = 1
num_classes = 2
epoch = 20
batch_size = 20

apply_DNN_BigData("minmax", num_classes, num_dataset, epoch, batch_size)
#apply_DNN_SmallData("minmax", num_classes, num_dataset, epoch, batch_size)