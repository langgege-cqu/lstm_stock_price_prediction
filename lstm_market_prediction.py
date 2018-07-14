import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

path_to_set = '[[PATH TO YOUR STOCK PRICE DATASET]]'


dataset = pd.read_csv(path_to_set, encoding='utf-8', index_col='date')
dataset.index = pd.to_datetime(dataset.index)
dataset = dataset.dropna(axis='columns')
stock_list = dataset.columns


np.random.seed(7)
step = 1
look_back = 240
train_size = 0.80


dataset = dataset.pct_change()
dataset = dataset[1:]
dataset['mean'] = dataset.mean(axis=1)


for c in dataset.columns:
    dataset[c + '_out'] = np.where(dataset[c] >= dataset['mean'], 0, 1)
    dataset[c] = (dataset[c] - dataset[c][:int(len(dataset) * train_size)].mean())/dataset[c][:int(len(dataset) * train_size)].std()

trainset = dataset[:int(len(dataset) * train_size)]
testset = dataset[int(len(dataset) * (train_size)):]


X_s = np.empty((0, look_back, step))
y_s = np.empty((0, 2))

for stock in stock_list:
    timeseries = np.asarray(trainset[stock])
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T

    X = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))

    y_series = np.asarray(trainset[stock + '_out'])
    y = y_series[look_back:]
    y = np_utils.to_categorical(y)

    X_s = np.append(X_s, X, axis=0)
    y_s = np.append(y_s, y, axis=0)


model = Sequential()
model.add(LSTM(25, input_shape=(240, 1)))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss="binary_crossentropy", optimizer="rmsprop")
model.fit(X_s, y_s, epochs=1000, batch_size=250, verbose=1, shuffle=False, callbacks=[EarlyStopping(patience=10)])


model_json = model.to_json()
with open("LSTM norm.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("LSTM norm weights.h5")
print("Saved!!!!")


predictors = stock_list

for c in predictors:
    testset[c + '_dn'] = 0.0000000
    testset[c + '_up'] = 0.0000000
    for i in range(len(testset.index)):
        if i > look_back:
            b = testset.loc[testset.index[i - look_back:i], c].as_matrix()
            yp = model.predict(b.reshape(1, look_back, 1))
            testset.loc[testset.index[i], c + '_dn'] = yp[0][0]
            testset.loc[testset.index[i], c + '_up'] = yp[0][1]

