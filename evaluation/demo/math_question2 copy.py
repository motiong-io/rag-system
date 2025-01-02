from scipy.optimize import minimize
import numpy as np

# 定义目标函数
def f(x):
    return x[0]**3 - 3 * x[1] * x[0]**2 + 2 * x[0]

# 初始点
x0 = np.array([0, 0])
bounds = [(0, 2),(-1, 1)]

# 优化
result = minimize(f, x0, method='L-BFGS-B', bounds=bounds)

# 打印结果
print( result)
