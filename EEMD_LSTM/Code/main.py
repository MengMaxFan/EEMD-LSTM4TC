import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from PyEMD import EEMD

from utils import data_split,data_split_LSTM,imf_data,visualize,LSTM_Model,RMSE,MAPE,calc_corr


    
plt.rcParams['figure.figsize'] = (10.0, 5.0)  
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

dataset = pd.read_csv('../csv/data_final.csv', header=0, index_col=0, parse_dates=True)
data = dataset.values.reshape(-1)

values = dataset.values
groups = [0, 1, 2, 3]

df = pd.DataFrame(dataset)  
df  = df.fillna(df.interpolate())


do = df['waveheight'][:]


# MinMaxScaler
DO = []
for i in range(0, len(do)):
    DO.append([do[i]])
scaler_DO = MinMaxScaler(feature_range=(0, 1))
DO = scaler_DO.fit_transform(DO)


# set IMFs = 8
eemd = EEMD()
eemd.noise_seed(12345)
imfs = eemd.eemd(DO.reshape(-1),None,8)

c = int(len(DO) * .85)

# Look back
lookback_window = 10
imfs_prediction = []

i = 1
for imf in imfs:
   plt.subplot(len(imfs), 1, i)
   plt.plot(imf)
   i += 1

plt.savefig('result_imf.png')
plt.show()



# Lead time = 30min*Pre_step ,if prtd
PRE_STEP = 2
test = np.zeros([len(DO) - c - lookback_window - PRE_STEP, 1])

# Train EEMD_LSTM 
i = 1
for imf in imfs:
    print('-'*45)
    print('This is  ' + str(i)  + '  time(s)')
    print('*'*45)
    X1_train, Y1_train, X1_test, Y1_test = data_split(imf_data(imf,1), c, lookback_window,PRE_STEP)
    X2_train, Y2_train, X2_test, Y2_test = data_split_LSTM(X1_train, Y1_train, X1_test, Y1_test)
    test += Y2_test
    model = LSTM_Model(X2_train,Y2_train,i)
#     model.save('../lbw6/EEMD-LSTM-imf' + str(i) + '-100.h5')
    prediction_Y = model.predict(X2_test)
    imfs_prediction.append(prediction_Y)
    i+=1;


imfs_prediction = np.array(imfs_prediction)
prediction = [0.0 for i in range(len(test))]
prediction = np.array(prediction)
for i in range(len(test)):
    t = 0.0
    for imf_prediction in imfs_prediction:
        t += imf_prediction[i][0]
    prediction[i] = t

prediction = prediction.reshape(prediction.shape[0], 1)

test = scaler_DO.inverse_transform(test)
prediction = scaler_DO.inverse_transform(prediction)


# EVALUATION
rmse=format(RMSE(test,prediction),'.4f')
mape=format(MAPE(test,prediction),'.4f')
r2 = format(r2_score(test, prediction), '.4f')
mae = format(mean_absolute_error(test, prediction), '.4f')
cor2 = calc_corr(test, prediction)
print('RMSE:' + str(rmse) + '\n' +  'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2)  + '\n' + 'r:' + cor2)




# PLOT
fig = plt.figure(figsize=(20,8))
ax = plt.gca()
plt.plot(test)
plt.plot(prediction, color='red')
ax.legend(['Wave Height', 'Forecast'])





# LSTM  Model

X1_train_lstm, Y1_train_lstm, X1_test_lstm, Y1_test_lstm = data_split(DO, c, lookback_window,PRE_STEP)

X2_train_lstm, Y2_train_lstm, X2_test_lstm, Y2_test_lstm = data_split_LSTM(X1_train_lstm, Y1_train_lstm, X1_test_lstm, Y1_test_lstm)

model = LSTM_Model(X2_train_lstm,Y2_train_lstm,1)
prediction_Y_lstm = model.predict(X2_test_lstm)
prediction_Y_lstm = np.array(prediction_Y_lstm)

prediction_lstm = scaler_DO.inverse_transform(prediction_Y_lstm)

rmse_LSTM=format(RMSE(test,prediction_lstm),'.4f')
mape_LSTM=format(MAPE(test,prediction_lstm),'.4f')
r2_LSTM = format(r2_score(test, prediction_lstm), '.4f')
mae_LSTM = format(mean_absolute_error(test, prediction_lstm), '.4f')
cor2_LSTM = calc_corr(test, prediction_lstm)
print('LSTM:' +'\n'+'RMSE:' + str(rmse_LSTM) + '\n' +  'MAE:' + str(mae_LSTM) + '\n' + 'MAPE:' + str(mape_LSTM) + '\n' + 'R2:' + str(r2_LSTM)+ '\n' + 'r:' + cor2_LSTM)

