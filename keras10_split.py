from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4, shuffle=False)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

# 3. 훈련

model.compile(loss='mse', optimizer='adam',
            #   metrics=['accuracy'])
              metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE( y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


# mse :  0.0002495874068699777
# [[80.98718 ]
#  [81.98688 ]
#  [82.98657 ]
#  [83.986275]
#  [84.985954]
#  [85.985664]
#  [86.98535 ]
#  [87.985054]
#  [88.98476 ]
#  [89.98444 ]
#  [90.984146]
#  [91.98383 ]
#  [92.98355 ]
#  [93.98326 ]
#  [94.98293 ]
#  [95.982635]
#  [96.98233 ]
#  [97.98203 ]
#  [98.98172 ]
#  [99.98141 ]]
# RMSE :  0.015797872895296355
# R2 :  0.9999924940514883