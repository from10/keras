from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.데이터
x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
            [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print("x.shape : ", x.shape)


# predict용 데이터

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1))) # 3 = 행 , 1 = 몇개씩 잘라서 작업
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)

x_input = array([25, 35, 45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input, batch_size=1)
print(yhat)

