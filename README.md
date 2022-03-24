# lstm_stock_model
## Introduction
An attempt on stock prediction using simple lstm model

## Data used
Closing price of the stock is used to fit in the model

## Train Test Split
80% training : 20% testing is used

## Model
LSTM(50,
         activation='relu',
         #return_sequences=True,
         input_shape=(look_back, 1))
         
Dropout(0.2)

Dense(units=1)

optimizer='adam', loss='mse'

## Training details
num_epochs = 25
