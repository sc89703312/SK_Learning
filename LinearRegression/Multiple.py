# 使用了 Batch Gradient Descent 的方式进行多元变量的线性回归分析
# 与单元变量的线性回归分析不同的是 需要将每个变量投影到合适的区域内

import numpy as np
import matplotlib.pyplot as plt

filename = 'ex1data2.txt'
data = np.loadtxt(filename, delimiter=',')

x_arr = data[:, :-1]

m = len(x_arr)
n = len(x_arr[0])

normal_arr = []

y_arr = data[:, -1:]


x_input = np.ones(m)

# 正则化各个参数
for i in range(n):
    x_i_arr = data[:, i]
    x_i_arr_mean = np.mean(x_i_arr)
    x_i_arr_std = np.std(x_i_arr)
    normal_arr.append([x_i_arr_mean, x_i_arr_std])
    x_i_arr = (x_i_arr - x_i_arr_mean) / x_i_arr_std

    x_input = np.vstack([x_input, x_i_arr])

x_input = x_input.T

iteration_arr = []
cost_arr = []

np.random.seed(0)
theta = np.zeros(n+1)
error = np.zeros(n+1)

alpha = 0.01
epsilon = 1e-3
max_loop = 10000
count = 0

while count < max_loop:
    count = count + 1

    diff_sum = np.zeros(n+1)
    for i in range(m):
        diff = ((np.dot(theta, x_input[i]) - y_arr[i]) * x_input[i])
        diff_sum += diff

    theta = theta - (alpha/m) * diff_sum

    # 计算每次的 cost
    cost_vector = np.matmul(x_input, theta.reshape(1, n+1).T) - y_arr
    cost = np.dot(cost_vector.T, cost_vector) / (2*m)
    iteration_arr.append(count)
    cost_arr.append(cost[0][0])

    # 结束条件
    if np.linalg.norm(theta - error) < epsilon:
        break
    else:
        error = theta
        pass

print(theta)
print(normal_arr)
plt.plot(iteration_arr, cost_arr, 'b')
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()
