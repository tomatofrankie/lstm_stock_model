import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import plotly.graph_objs as go
from keras.preprocessing.sequence import TimeseriesGenerator

filename = "stock_data/GME.csv"

df = pd.read_csv(filename)
#print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

close_data = df['Close'].values
close_data = close_data.reshape((-1, 1))

split_percent = 0.8
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

# close_date = df['Date']
# close_train,close_test,date_train,date_test = train_test_split(close_data,close_date,test_size=0.2)


look_back = 10

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=10)
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(
    LSTM(50,
         activation='relu',
         #return_sequences=True,
         input_shape=(look_back, 1))
)
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
#model.save('0302_model_30.h5')

prediction = model.predict_generator(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x=date_train,
    y=close_train,
    mode='lines',
    name='Data'
)
trace2 = go.Scatter(
    x=date_test,
    y=prediction,
    mode='lines',
    name='Prediction'
)
trace3 = go.Scatter(
    x=date_test,
    y=close_test,
    mode='lines',
    name='Ground Truth'
)
layout = go.Layout(
    title="Stock",
    xaxis={'title': "Date"},
    yaxis={'title': "Close"}
)

close_data = close_data.reshape((-1))


def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

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


num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
print(forecast)
trace4 = go.Scatter(
    x=forecast_dates,
    y=forecast,
    mode='lines',
    name='Future Prediction'
)
fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
fig.show()
