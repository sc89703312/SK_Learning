# 使用了 Normal Equations 的方式进行多元变量的线性回归分析
# theta = (X^T X)^-1 X^T y


import numpy as np

filename = 'ex1data2.txt'
data = np.loadtxt(filename, delimiter=',')

x_arr = data[:, :-1]
y_arr = data[:, -1:]

m = len(x_arr)
n = len(x_arr[0])

x0 = np.full(m, 1.0).reshape(m, 1)
x_input = np.hstack([x0, x_arr])

theta = np.linalg.pinv(x_input.T.dot(x_input)).dot(x_input.T.dot(y_arr))

print(theta)

