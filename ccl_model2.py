from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def predict(num_prediction, model):
    prediction_list = new_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]

    return prediction_list


def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates


look_back = 200
model = load_model('saved_lstm_model_ccl.h5')
df = pd.read_csv("T2.csv")

df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'High'])

new_data = df['Close'].values
new_data = new_data.reshape((-1, 1))

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
print(forecast)
print(forecast_dates)
plt.figure(figsize=(16, 8))
plt.plot(forecast_dates, forecast)

df.head()

df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

plt.plot(df["High"], label='High Price history')

new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'High'])

for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["High"][i] = data["High"][i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

final_dataset = new_dataset.values

train_data = final_dataset[0:1000, :]
valid_data = final_dataset[1000:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(200, len(train_data)):
    x_train_data.append(scaled_data[i - 30:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 200:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test = []
for i in range(200, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 200:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train_data = new_dataset[:1000]
# valid_data = new_dataset[202:]
# valid_data['Predictions'] = closing_price
plt.plot(train_data["High"])
# plt.plot(valid_data[['High', "Predictions"]])
plt.show()
