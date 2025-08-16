import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


model = load_model(r'C:\ml files\corizo_mini_project\Stock Predictions Model.keras')

st.header('Stock Market Predictor')


stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'


data = yf.download(stock, start, end)
st.subheader('stock Data')
st.write(data)




data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test=pd.concat([pas_100_days,data_train],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('prive vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('prive vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('prive vs MA50 va MA100')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)

x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


predict = model.predict(x_test)


scale_factor = 1 / scaler.scale_[0]
redict = predict * scale_factor
y_test = y_test * scale_factor


import numpy as np

st.subheader('Prediction vs Original')

# Flatten or reshape predictions to 1D
predict_1d = np.array(predict).reshape(predict.shape[0], -1)[:, -1]  # Take last timestep
y_test_1d = np.array(y_test).reshape(y_test.shape[0], -1)[:, -1]     # Take last timestep

# Plot
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y_test_1d, 'g', label='Original Price')      # Actual values
plt.plot(predict_1d, 'r', label='Predicted Price')    # Predictions
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Show in Streamlit


st.pyplot(fig4)
