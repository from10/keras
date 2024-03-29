from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])


x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)


print(x1.shape)
print(y2.shape) 
print(y3.shape) 

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=66, test_size=0.4, shuffle=False
    )

x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state=66, test_size=0.5, shuffle=False
    ) 

y2_train, y2_test = train_test_split(
    y3, random_state=66, test_size=0.4
    )

y2_val, y2_test = train_test_split(
    y3_test, random_state=66, test_size=0.5
    ) 

print(y3_test.shape) # (20, 3)

# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(1,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
middle1 = Dense(2)(dense3)

input2 = Input(shape=(1,))
xx = Dense(5, activation='relu')(input2)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
middle2 = Dense(2)(xx)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

output2 = Dense(30)(merge1)
output2 = Dense(10)(output2)
output2 = Dense(2)(output2)


model = Model(inputs = input1, outputs = [output1, output2])
model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam',
            #   metrics=['accuracy'])
              metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x1_train, [y1_train, y2_train], epochs=150, batch_size=1,
          validation_data=(x1_val, [y1_val, y2_val]))
    

# 4. 평가 예측
mse = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
print("mse : ", mse[0])
print("mse : ", mse[1])
print("mse : ", mse[2])


y1_predict, y2_predict = model.predict(x1_test) 
print(y1_predict)
print(y2_predict)


'''
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE( y_test, y_predict ):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
'''

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE( xxx, yyy ):
    return np.sqrt(mean_squared_error(xxx, yyy))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2) / 2)


# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)

print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict) / 2 )
