# loss, acc, val_loss, val_acc

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_predict = np.array([21, 22, 23, 24, 25])

x2 = np.array([11, 12, 13, 14, 15])

model = Sequential()

model.add(Dense(500, input_shape=(1, ), activation='relu'))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # acc라고 써도 됨
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=5000, batch_size=1, verbose=2, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_predict)
print(y_predict)
