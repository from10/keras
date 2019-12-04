from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(7, input_shape=(1, ), activation='relu'))
model.add(Dense(5))
model.add(Dense(3))
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


# mse :  5.739508424085216e-07
# [[80.999374]
#  [81.99936 ]
#  [82.99934 ]
#  [83.99932 ]
#  [84.99931 ]
#  [85.999306]
#  [86.99928 ]
#  [87.999275]
#  [88.99926 ]
#  [89.99925 ]
#  [90.99924 ]
#  [91.99923 ]
#  [92.999214]
#  [93.99921 ]
#  [94.99919 ]
#  [95.99917 ]
#  [96.99915 ]
#  [97.999146]
#  [98.99914 ]
#  [99.999115]]
# RMSE :  0.0007598011319902586
# R2 :  0.9999999826376613