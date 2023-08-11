#############################################
## this file is to build the lstm model, and save the mdoel,
## which is used for map preiction
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
tf.random.set_seed(1)
np.random.seed(1)


if __name__ == "__main__":
    # path = "C:/Users/xmblb/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/"
    path = "C:/Users/DELL/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/"
    start_day = 0  # a value to determine the length of rainfall curve
    random_seed = 1 # set a seed for repeated randomly spliting

    ##########################################
    # ########
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
    all_data_neg = pd.read_csv(path + "sample absence.csv") # read landslide presence samples
    ## remove the totall same precipitation negative curves
    all_data_neg = all_data_neg.drop_duplicates(subset=all_data_neg.columns[start_day:60], keep="first")

    ## keep the absence curves that any daily prec value above the pos threshold
    retain_neg_data = []
    for single_neg in all_data_neg.values:
        temp = single_neg[start_day:60] > threshold
        if True in temp:
            retain_neg_data.append(single_neg)
    all_data_neg = pd.DataFrame(retain_neg_data).values
    # temp = all_data_neg.values
    # index_neg = np.argwhere(all_data_neg[:, 59] != 0)

    # ## remove that the rainfall in 60th day is equal to 0
    # index_neg = np.argwhere(all_data_neg[:,59]!=0) ## find the index that the rainfall in 60th day above 0
    # index_neg = np.ravel(index_neg)
    # all_data_neg = all_data_neg[index_neg,:] ## get the data
    data_neg_data, nonlandslide_test = train_test_split(all_data_neg, train_size=data_pos_data.shape[0],shuffle=True, random_state=1)
    #######################################################################


    ######################################################
    # divide the model data into training and testing sets
    ######################################################
    ## randomly select 70% data for modelling, the leat-out for testing

    # merge the train pos and train neg curves
    train_data = np.concatenate((data_pos_data, data_neg_data), axis=0)

    train_x, train_y_1D = train_data[:, start_day:-3], train_data[:, -3]
    train_x = np.expand_dims(train_x, axis=2)
    train_x = train_x.astype(np.float32)
    train_y_1D = train_y_1D.astype(np.int)
    train_y = np_utils.to_categorical(train_y_1D, 2)
    ###################################################

    ############################################
    # build the LSTM model
    ##########################################
    model = Sequential()
    model.add(LSTM(16, input_shape=(60 - start_day, 1),  activation='tanh'))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    opt = adam.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = loss_history.LossHistory()
    # train the LSTM model
    model.fit(train_x, train_y, validation_data=(train_x, train_y), callbacks=[history], shuffle=True,
              verbose=2, batch_size=64, epochs=30)
    history.loss_plot('epoch')
    ########################################################################################

    model.save(path+'LSTM fitting model1.h5')
    # from keras.models import load_model
    # model = load_model(path+'Best LSTM model.h5')


    # model validaton
    y_prob_test = model.predict(train_x)  # output predict probability
    y_probability_first = [prob[1] for prob in y_prob_test]

    y_pred_label = [1 if data >= 0.5 else 0 for data in y_probability_first ]
    test_acc = metrics.accuracy_score(train_y_1D, y_pred_label)
    test_auc = metrics.roc_auc_score(train_y_1D, y_probability_first)
    print(test_acc, test_auc)


    ## plot the ROC curves
    fpr, tpr, thresholds = roc_curve(train_y_1D, y_probability_first)
    plt.figure()
    plt.plot(fpr, tpr)

    index_neg = np.argwhere(data_neg_data.values[:, 59] != 0)