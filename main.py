import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2

def grad_f(x):
    return np.array([2 * x[0] + 2 * x[1], 2 * x[1] + 6 * x[1]])

def hessian_f(x):
    return np.array([[2+2*x[1], 2*x[0]], [2*x[0], 6+6*x[1]]])

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