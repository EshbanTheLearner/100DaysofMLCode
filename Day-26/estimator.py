print('Importing Libraries...\n')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
print('Done!\n')

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(101)

print('Loading Data...\n')
# load the dataset
dataframe = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
print('Done!\n')
dataset = dataframe.values
dataset = dataset.astype('float32')

print('Here! Have a look!')
# visualising the dataset
plt.plot(dataset)
plt.show()

print('Scaling Data...\n')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print('Done!\n')

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print('Building our estimator...\n')
# create and fit the LSTM network
estimator = Sequential()
estimator.add(LSTM(4, input_shape=(1, look_back)))
estimator.add(Dense(1))

# compiling our estimator
estimator.compile(loss='mean_squared_error', optimizer='adam')
print('Done!\n')
estimator.summary()

estimator.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)


# make predictions
pred_train = estimator.predict(X_train)
pred_test = estimator.predict(X_test)

# invert predictions
pred_train = scaler.inverse_transform(pred_train)
y_train = scaler.inverse_transform([y_train])
pred_test = scaler.inverse_transform(pred_test)
y_test = scaler.inverse_transform([y_test])

print('Calculating RMSE...\n')
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], pred_train[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], pred_test[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
pred_plot_train = np.empty_like(dataset)
pred_plot_train[:, :] = np.nan
pred_plot_train[look_back:len(pred_train)+look_back, :] = pred_train

# shift test predictions for plotting
pred_plot_test = np.empty_like(dataset)
pred_plot_test[:, :] = np.nan
pred_plot_test[len(pred_train)+(look_back*2)+1:len(dataset)-1, :] = pred_test

print('Plotting predictions...\n')
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(pred_plot_train)
plt.plot(pred_plot_test)
plt.show()
print('Done! \nThanks!')