from pandas_datareader.data import DataReader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Get the stock quote

while True:
    user_input = input('Input the ticker:')
    if user_input == 'quit()':
        break
    df = DataReader(user_input, data_source='yahoo', start='2018-01-01', end=datetime.now())
    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    print(data)

    # Convert the dataframe to a numpy array
    dataset = data.values
    date = data.index

    date = np.linspace(1,len(date),len(date))
    dataset = dataset.flatten()
    mymodel = np.poly1d(np.polyfit(date, dataset, 3))
    plt.figure(figsize=(12, 10), dpi=80)
    plt.scatter(date, dataset)
    plt.plot(date, mymodel(date))
    plt.show()

