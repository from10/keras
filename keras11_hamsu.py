from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4, shuffle=False)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2


# 2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

# input1 = Input(shape=(1,))
# dense1 = Dense(10, activation='relu')(input1)
# dense2 = Dense(9)(dense1)
# dense3 = Dense(8)(dense2)
# dense4 = Dense(7)(dense3)
# dense5 = Dense(6)(dense4)
# dense6 = Dense(5)(dense5)
# dense7 = Dense(4)(dense6)
# dense8 = Dense(3)(dense7)
# dense9 = Dense(2)(dense8)
# dense10 = Dense(1)(dense9)
# output1 = Dense(1)(dense10)

input1 = Input(shape=(1,))
xx = Dense(10, activation='relu')(input1)
xx = Dense(9)(xx)
xx = Dense(8)(xx)
xx = Dense(7)(xx)
xx = Dense(6)(xx)
xx = Dense(5)(xx)
xx = Dense(4)(xx)
xx = Dense(3)(xx)
xx = Dense(2)(xx)
xx = Dense(1)(xx)
output1 = Dense(1)(xx)



model = Model(inputs = input1, outputs = output1)
model.summary()

# 레이어를 10개 이상 늘리시오


