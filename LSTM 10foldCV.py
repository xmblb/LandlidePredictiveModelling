#############################################
## this file is to build the lstm model based on balanced dataset
## for the 2380 landslide locations, we extract the 2380 curves of specific time as presence data
## this file uses a 10-fold cross-validation for validation
################################################
import keras
import numpy as np
import pandas as pd
from keras.optimizer_v2 import adam, adamax, adagrad, adadelta
from common_func import loss_history
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(1)


if __name__ == "__main__":
    path = "C:/Users/DELL/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/"
    start_day = 0  # a value to determine the length of rainfall curve
    ##################################################
    ### positive data preprocessing
    ##################################################
    data_pos_data = pd.read_csv(path + "sample presence.csv") # read landslide presence samples
    data_pos_data = data_pos_data.values
    index = np.argwhere(data_pos_data[:,59]!=0) ## find the index that the rainfall in 60th day above 0
    index = np.ravel(index)
    data_pos_data = data_pos_data[index,:] ## get the data
    ## remove the totall same precipitation curves
    data_pos_data = pd.DataFrame(data_pos_data)
    data_pos_data = data_pos_data.drop_duplicates(subset=data_pos_data.columns[start_day:60], keep="first")

    ## get the threshold to select negtative curves (mean + 2sd)
    all_pos_mean = data_pos_data.iloc[:,start_day:60].mean(axis = 1).values
    pos_mean, pos_sd = np.mean(all_pos_mean), np.std(all_pos_mean)
    threshold = pos_mean + 2*pos_sd
    ###############################################################################


    ##################################################
    ### negative data preprocessing
    ##################################################
    data_pos_data1 = pd.read_csv(path + "sample absence1.csv")
    data_pos_data2 = pd.read_csv(path + "sample absence2.csv")
    data_pos_data3 = pd.read_csv(path + "sample absence3.csv")
    data_pos_data4 = pd.read_csv(path + "sample absence4.csv")
    data_pos_data5 = pd.read_csv(path + "sample absence5.csv")
    data_pos_data6 = pd.read_csv(path + "sample absence6.csv")
    data_pos_data7 = pd.read_csv(path + "sample absence7.csv")
    data_pos_data8 = pd.read_csv(path + "sample absence8.csv")
    data_pos_data9 = pd.read_csv(path + "sample absence9.csv")
    data_pos_data10 = pd.read_csv(path + "sample absence10.csv")
    all_data_neg = pd.concat([data_pos_data1, data_pos_data2, data_pos_data3,data_pos_data4,data_pos_data5,data_pos_data6,
                              data_pos_data7,data_pos_data8,data_pos_data9,data_pos_data10], axis=0)
    ## remove the totall same precipitation negative curves
    all_data_neg = all_data_neg.drop_duplicates(subset=all_data_neg.columns[start_day:60], keep="first")

    ## keep the absence curves that any daily prec value above the pos threshold
    retain_neg_data = []
    for single_neg in all_data_neg.values:
        temp = single_neg[start_day:60] > threshold
        if True in temp:
            retain_neg_data.append(single_neg)
    all_data_neg = pd.DataFrame(retain_neg_data).values

    ##############################################################################
    # set a seed for repeated randomly sampling
    # a loop to evaulate the sensitivity of randomly sampling
    ######################################################################
    all_auc, all_acc, all_recall, all_precision, all_f1, all_kappa = [], [], [], [], [], []
    for random_seed in range(1, 11):
        # randomly selected negative samples from the large absence set
        data_neg_data, nonlandslide_test = train_test_split(all_data_neg, train_size=data_pos_data.shape[0], shuffle=True, random_state = random_seed)

        ######################################################
        # 10 fold cross validation
        ######################################################
        all_data = np.concatenate((data_pos_data, data_neg_data), axis=0)
        inputs = all_data[:,start_day:60]
        inputs = np.expand_dims(inputs, axis=2)
        inputs = inputs.astype(np.float32)
        # the label
        targets = all_data[:,60]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1)
        fold_number = 1
        temp_auc, temp_acc, temp_recall, temp_precision, temp_f1, temp_kappa = [], [], [], [], [], []
        for train_idx, test_idx in kfold.split(inputs, targets):
            tf.random.set_seed(1)
            # define the lstm model
            model = Sequential()
            model.add(LSTM(16, input_shape=(60 - start_day, 1), activation='tanh'))
            model.add(Dense(2, activation='softmax'))
            print(model.summary())
            opt = adam.Adam()
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # train the LSTM model
            train_y_1D = targets[train_idx].astype(np.int)
            train_y = np_utils.to_categorical(train_y_1D, 2) # the label of training data
            test_y_1D = targets[test_idx].astype(np.int)
            test_y = np_utils.to_categorical(test_y_1D, 2) # the label of test data
            model.fit(inputs[train_idx], train_y, validation_data=(inputs[test_idx], test_y), shuffle=True,
                      verbose=2, batch_size=64, epochs=30)

            # evaluate the model
            y_prob_test = model.predict(inputs[test_idx])  # output predict probability
            y_probability_first = [prob[1] for prob in y_prob_test]

            y_pred_label = [1 if data >= 0.5 else 0 for data in y_probability_first]
            test_auc = metrics.roc_auc_score(test_y_1D, y_probability_first)
            test_acc = metrics.accuracy_score(test_y_1D, y_pred_label)
            test_recall = metrics.recall_score(test_y_1D, y_pred_label)
            test_precision = metrics.precision_score(test_y_1D, y_pred_label)
            test_f1 = metrics.f1_score(test_y_1D, y_pred_label)
            test_kappa = metrics.cohen_kappa_score(test_y_1D, y_pred_label)
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_number} ...')
            # print(f'Training for fold {fold_number} ...')
            print(test_acc, test_auc, test_recall, test_precision, test_f1, test_kappa)
            fold_number += 1
            temp_auc.append(np.round(test_auc, 3))
            temp_acc.append(np.round(test_acc, 3))
            temp_recall.append(np.round(test_recall, 3))
            temp_precision.append(np.round(test_precision, 3))
            temp_f1.append(np.round(test_f1, 3))
            temp_kappa.append(np.round(test_kappa, 3))

        all_auc.append(temp_auc)
        all_acc.append(temp_acc)
        all_recall.append(temp_recall)
        all_precision.append(temp_precision)
        all_f1.append(temp_f1)
        all_kappa.append(temp_kappa)

    all_auc, all_acc, all_recall, all_precision, all_f1, all_kappa = \
        pd.DataFrame(all_auc), pd.DataFrame(all_acc), pd.DataFrame(all_recall), pd.DataFrame(all_precision), pd.DataFrame(all_f1), pd.DataFrame(all_kappa)
    all_auc.to_csv(path + 'result_auc.csv')
    all_acc.to_csv(path + 'result_acc.csv')
    all_recall.to_csv(path + 'result_recall.csv')
    all_precision.to_csv(path + 'result_precision.csv')
    all_f1.to_csv(path + 'result_f1.csv')
    all_kappa.to_csv(path + 'result_kappa.csv')
    ########################################################################################


