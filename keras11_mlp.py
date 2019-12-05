# 1. 데이터
import numpy as np

# x = np.array([range(1,101))
# y = np.array([range(1,101))
x = np.array([range(1,101), range(101, 201)])
y = np.array([range(1,101), range(101, 201)])
# print(x)

print(x.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)

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
model.add(Dense(5, input_shape=(2, ), activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))

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



# mse :  0.0001228972541866824
# [[ 80.992294 180.98828 ]
#  [ 81.9922   181.98816 ]
#  [ 82.99212  182.98802 ]
#  [ 83.99201  183.98785 ]
#  [ 84.99192  184.98772 ]
#  [ 85.99182  185.98758 ]
#  [ 86.99172  186.98741 ]
#  [ 87.99163  187.98729 ]
#  [ 88.99154  188.98714 ]
#  [ 89.99144  189.987   ]
#  [ 90.99137  190.98688 ]
#  [ 91.99126  191.98672 ]
#  [ 92.99115  192.98656 ]
#  [ 93.991066 193.98643 ]
#  [ 94.99097  194.98627 ]
#  [ 95.99087  195.98611 ]
#  [ 96.99079  196.98602 ]
#  [ 97.990685 197.98586 ]
#  [ 98.99059  198.9857  ]
#  [ 99.99049  199.98555 ]]
# RMSE :  0.011087817734946924
# R2 :  0.9999963025653497

