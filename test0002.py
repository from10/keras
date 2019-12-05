import numpy as np

aaa = np.array([1, 2, 3, 4, 5])
print(aaa.shape) # (5, ) ≒ (1,5)

aaa = aaa.reshape(1, 5)
print(aaa.shape) # (1,5)

bbb = np.array([[1, 2, 3], [4, 5, 6]])
print(bbb.shape) # (2,3)

ccc = np.array([[1, 2], [3, 4], [5, 6]])
print(ccc.shape) # (3,2)

ddd = ccc.reshape(3,2,1,1)
print(ddd.shape)

print(ddd)

