import numpy as np
import pandas as pd


from pandas import read_csv
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot
from pylab import mpl


from pandas import concat
from PyEMD import EEMD

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Activation

from scipy import interpolate, math
import matplotlib.pyplot as plt

from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# from sklearn.ensemble import IsolationForest


def data_split(data, train_len, lookback_window,PRE_STEP):

    WINDOW_SIZE = 10
    train = data[:train_len]  
    test = data[train_len:]  

    features = []
    predict = []
    for i in range(len(train) - WINDOW_SIZE - PRE_STEP):
        end_ix = i + WINDOW_SIZE
        out_end_ix = end_ix + PRE_STEP
        x = train[i:end_ix]
        y = train[out_end_ix]
        features.append(x)
        predict.append(y)
    features = np.array(features)
    predict = np.array(predict)
    
    
    features1 = []
    predict1 = []
    for i in range(len(test) - WINDOW_SIZE - PRE_STEP):
        end_ix = i + WINDOW_SIZE
        out_end_ix = end_ix + PRE_STEP
        x = test[i:end_ix]
        y = test[out_end_ix]
        features1.append(x)
        predict1.append(y)
    features1 = np.array(features1)
    predict1 = np.array(predict1)

    return (features,predict, features1,predict1)



def data_split_LSTM(X_train, Y_train, X_test, y_test):  # data split f
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    return (X_train, Y_train, X_test, y_test)


def imf_data(data, lookback_window):
    X1 = []
    for i in range(lookback_window, len(data)):
        X1.append(data[i - lookback_window:i])
    X1.append(data[len(data) - 1:len(data)])
    X_train = np.array(X1)
    return X_train


def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def LSTM_Model(X_train, Y_train,i):

    model = Sequential()
    model.add(LSTM(50,activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))  
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=2, shuffle=False)
    return (model)


def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse


def MAPE(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.fabs((Y_true - Y_pred) / Y_true)) * 100

def calc_corr(a, b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))

    corr_factor = cov_ab/sq

    return corr_factor