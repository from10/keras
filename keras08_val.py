from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
import numpy as np
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 107, 1032, 1042, 1052])

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


# mse :  0.0022018395829945803
# [[10.979211 ]
#  [11.973958 ]
#  [12.968703 ]
#  [13.9634495]
#  [14.958196 ]
#  [15.952943 ]
#  [16.94769  ]
#  [17.942436 ]
#  [18.937181 ]
#  [19.931927 ]]
# RMSE :  0.04692327493997024
# R2 :  0.9997331159113828