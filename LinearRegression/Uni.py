import numpy as np
import matplotlib.pyplot as plt

# 使用了 Batch Gradient Descent 的方式进行一元变量的线性回归分析

filename = 'ex1data1.txt'
data = np.loadtxt(filename, delimiter=',')

x_arr = data[:, 0]
y_arr = data[:, -1:]
iteration_arr = []
cost_arr = []

x_input = x_arr

# 给X的第一列补上 全1列，便于计算
m = len(x_arr)
x0 = np.full(m, 1.0)
x_arr = np.vstack([x0, x_arr]).T

np.random.seed(0)
theta = np.random.randn(2)
error = np.zeros(2)

alpha = 0.01
epsilon = 1e-3
max_loop = 10000
count = 0

while count < max_loop:
    count = count + 1

    diff_sum = np.zeros(2)
    for i in range(m):
        diff = ((np.dot(theta, x_arr[i]) - y_arr[i]) * x_arr[i])
        diff_sum += diff

    theta = theta - (alpha/m) * diff_sum

    # 计算每次的 cost
    cost_vector = np.matmul(x_arr, theta.reshape(1, 2).T) - y_arr
    cost = np.dot(cost_vector.T, cost_vector) / (2*m)
    iteration_arr.append(count)
    cost_arr.append(cost[0][0])

    # 结束条件
    if np.linalg.norm(theta - error) < epsilon:
        break
    else:
        error = theta

plt.subplot(2, 1, 1)
plt.plot(x_input, y_arr, 'ro')
plt.plot(x_input, theta[0] + theta[1] * x_input, 'g')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2, 1, 2)
plt.plot(iteration_arr, cost_arr, 'b')
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()
