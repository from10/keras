# 1. 데이터
import numpy as np

# x = np.array([range(1,101))
# y = np.array([range(1,101))
x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)])

# print(x)

print(x.shape)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=33, test_size=0.4, shuffle=False)  

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=33, test_size=0.5, shuffle=False) # 6:2:2

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5000, input_shape=(2, ), activation='relu'))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
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
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)

'''
aaa = np.array([[101, 102, 103], [201, 202, 203]])
aaa = np.transpose(aaa)
y_predict = model.predict(aaa) # x_test
'''

y_predict = model.predict(x_test) # x_test
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


   
# keras11_mlp2.py 카피
# R2 0.5 이하로 만들기
# 레이어는 5개 이상(히든만)
# 노드는 10개 이상
# epochs는 100개 이상
# batch_size=1 


# acc :  45.12407302856445
# [[285.93008]
#  [287.09912]
#  [288.2702 ]
#  [289.4437 ]
#  [290.61755]
#  [291.79364]
#  [292.97183]
#  [294.15067]
#  [295.33163]
#  [296.5147 ]
#  [297.70026]
#  [298.88712]
#  [300.07547]
#  [301.26474]
#  [302.45413]
#  [303.64526]
#  [304.837  ]
#  [306.03012]
#  [307.22412]
#  [308.42026]]
# RMSE :  6.71744131113473
# R2 :  -0.35711331634705235