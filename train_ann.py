import numpy as np
import pandas as pd
import math as math
# import pickle as pickle
import keras.optimizers as opt
from keras.models import Sequential

from keras import layers
from matplotlib import pyplot as plt


'''

    Data settings
    
'''

data_dir = 'data/'
file_name = 'RU_Electricity_Market_PZ_dayahead_price_volume.csv'
CUSTOM_DTYPES = {'price_eur': np.float32, 'price_sib': np.float32,
                 'consumption_eur': np.float32, 'consumption_sib': np.float32}

source_data = pd.read_csv(data_dir + file_name, parse_dates=['timestep'], dtype=CUSTOM_DTYPES)
source_data.index = source_data['timestep']
source_data.drop(columns=['timestep'], inplace=True)

target_name = 'consumption_sib'

'''

    Features engineering

'''

data = pd.DataFrame()
data[target_name] = source_data[target_name]
data['hour'] = data.index.hour.values
data['dayofweek'] = data.index.dayofweek.values
data['weekofyear'] = data.index.weekofyear.values
data['month'] = data.index.month.values
data['year'] = data.index.year.values

encoding_columns = ['hour', 'dayofweek']
for col in encoding_columns:
    data = pd.get_dummies(data, columns=[col], prefix=col)

sin_number = 5
for w in range(1, sin_number+1):
    data['sin_' + str(w)] = np.square(np.sin(data.index.dayofyear.astype('float64') / 365 * w * math.pi).values)

# plt.plot(data.index, data['sin_1'], label='sin_1')
# plt.plot(data.index, data['sin_5'], label='sin_5')
# plt.legend()
# plt.show()

lag_settings = [48, 72, 96, 120, 144]
col_names = ['lags_' + str(shift) for shift in lag_settings]
for idx in range(len(lag_settings)):
    data[col_names[idx]] = data[target_name].shift(lag_settings[idx])

# plt.plot(data.index, data[target_name], label=target_name)
# plt.plot(data.index, data['lags_72'], label='lag_72')
# plt.plot(data.index, data['lags_144'], label='lag_144')
# plt.legend()
# plt.show()

predictors_name = data.columns.drop(target_name)

data.dropna(inplace=True)
scaler_mean = data.values.mean(axis=0)
scaler_std = data.values.std(axis=0)
data_scaled = (data.values - scaler_mean) / scaler_std

print('Target: ', target_name)
print('Predictors (%s): ' % len(predictors_name), predictors_name)

predictors = data_scaled[:, data.columns.isin(predictors_name)]
target = data_scaled[:, data.columns.isin([target_name])]

'''

    Network settings

'''

neurons_number_layer = 48
train_split_index = 1400
batch_size = 30
network_epochs = 60
custom_learning_rate = 0.25e-3
custom_loss = 'mae'
horison = 24

predictors = predictors.reshape(-1, horison, predictors.shape[1])
target = target.reshape(-1, horison)
dates = data.index.values.reshape(-1, horison)

train_predictors = predictors[:train_split_index]
train_target = target[:train_split_index]
train_dates = dates[:train_split_index]

valid_predictors = predictors[train_split_index:]
valid_target = target[train_split_index:]
valid_dates = dates[train_split_index:]

network = Sequential()
network.add(layers.Flatten(input_shape=(horison, predictors.shape[-1])))
network.add(layers.Dense(neurons_number_layer, activation='selu'))
network.add(layers.Dense(horison))

custom_optimiser = opt.Adam(lr=custom_learning_rate)
network.compile(optimizer=custom_optimiser, loss=custom_loss)

history = network.fit(train_predictors, train_target,
                      epochs=network_epochs,
                      batch_size=batch_size,
                      validation_data=(valid_predictors, valid_target))


'''

    Assess train results
    
'''


network.summary()

loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


'''

    Prediction

'''

train_model = network.predict(train_predictors)
train_model = train_model.reshape(-1, 1)
train_target = train_target.reshape(-1, 1)

train_model = train_model * scaler_std[0] + scaler_mean[0]
train_target = train_target * scaler_std[0] + scaler_mean[0]

train_mae = np.mean(np.abs(train_model - train_target))
train_mape = np.mean(np.abs((train_target - train_model) / train_target)) * 100

print('Train: MAE = %.2f MWh, MAPE %.2f %%' % (train_mae, train_mape))

valid_model = network.predict(valid_predictors)
valid_model = valid_model.reshape(-1, 1)
valid_target = valid_target.reshape(-1, 1)

valid_model = valid_model * scaler_std[0] + scaler_mean[0]
valid_target = valid_target * scaler_std[0] + scaler_mean[0]

valid_mae = np.mean(np.abs(valid_model - valid_target))
valid_mape = np.mean(np.abs((valid_target - valid_model) / valid_target)) * 100

print('Valid: MAE = %.2f MWh, MAPE %.2f %%' % (valid_mae, valid_mape))


'''

    Plot

'''

mask_train = np.isin(dates.reshape(-1, 1), train_dates.reshape(-1, 1))
mask_valid = np.isin(dates.reshape(-1, 1), valid_dates.reshape(-1, 1))

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 20), sharex=False)
ax1.plot(data[mask_train].index, data.loc[mask_train, target_name], label=target_name)
ax1.plot(data[mask_train].index, train_model, label='train')
ax1.set_title('Train forecast %s, MAPE %.2f%%, MAE %.2f' % (target_name, train_mape, train_mae), fontsize=20)
ax1.set_ylabel('MWh')
ax1.legend()

ax2.plot(data[mask_valid].index, data.loc[mask_valid, target_name], label=target_name)
ax2.plot(data[mask_valid].index, valid_model, label='valid')
ax2.set_title('Valid forecast %s, MAPE %.2f%%, MAE %.2f' % (target_name, valid_mape, valid_mae), fontsize=20)
ax2.set_ylabel('MWh')
ax2.legend()
plt.subplots_adjust(hspace=0.3)
plt.show()

model = dict()
model['network'] = network
model['scaler'] = [scaler_mean, scaler_std]

# filename = 'models/' + target_name + '_ann.pickle'
# model_save = open(filename, 'wb')
# pickle.dump(model, model_save)
# model_save.close()
# print('Network model is saved in ' + filename)
# data.to_csv('data.csv')
