import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + x[1]**2 - 2*x[1] +x[0]**2
def grad_f(x):
    return np.array([2*x[0] + 2*x[0], 2*x[1] - 2])

#def f(x):
 #   return x[0]**2 + x[1]**2
#def grad_f(x):
  #  return np.array([2*x[0], 2*x[1]])

#def f(x):
 #   return x[0] ** 2 + x[1] ** 2 - 2 * x[1] + x[0] ** 2

#def grad_f(x):
 #   return np.array([2 * x[0] + 2 * x[0], 2 * x[1] - 2])

def hessian_f(x):
    h = 1e-5  # шаг для вычисления конечных разностей
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            # вычисляем конечную разность в направлении i-ой и j-ой координат
            # используем формулу (f(x + h*e_i + h*e_j) - f(x + h*e_i) - f(x + h*e_j) + f(x)) / h^2
            H[i, j] = (f(x + h*np.array([i == k for k in range(2)]) + h*np.array([j == k for k in range(2)])) - f(x + h*np.array([i == k for k in range(2)])) - f(x + h*np.array([j == k for k in range(2)])) + f(x)) / (h**2)
    return H

def steepest_descent(f, grad_f, hessian_f, x0, eps=1e-5, max_iter=100):
    x = x0
    grad = grad_f(x)
    d = -grad
    alpha = (grad.T @ grad) / (d.T @ hessian_f(x) @ d)
    x_list = [x]
    for i in range(max_iter):
        x_prev = x
        x = x + alpha*d
        grad_prev = grad
        grad = grad_f(x)
        beta = (grad.T @ grad) / (grad_prev.T @ grad_prev)
        d = -grad + beta*d
        alpha = (grad.T @ grad) / (d.T @ hessian_f(x) @ d)
        x_list.append(x)
        if np.linalg.norm(grad) < eps:
            break
    return x_list

x0 = np.array([-1, 0.0000001])
x_list = steepest_descent(f, grad_f, hessian_f, x0)

# plot function contours
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Contour plot')

# plot steepest descent iterations
x_list = np.array(x_list)
plt.plot(x_list[:, 0], x_list[:, 1], 'bo-')
plt.show()

print(x_list)