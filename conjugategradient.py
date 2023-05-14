import numpy as np
from scipy.optimize import brent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar

# определяем квадратичную функцию
def f(x):
    return x[0]**2 + 2 * x[1]**2

# определяем градиент функции
def grad_f(x):
    return np.array([2 * x[0], 4 * x[1]])

# реализуем метод сопряженных градиентов
def conjugate_gradient_method(x0, f, grad_f, epsilon=1e-5, max_iterations=1000, verbose=False):
    x = x0
    d = -grad_f(x)
    g = grad_f(x)
    alpha = 0.1
    x_hist = [x]

    for i in range(max_iterations):
        d_prev = d
        alpha = minimize_scalar(lambda a: f(x + a * d)).x
        x = x + alpha * d
        x_hist.append(x)
        g_prev = g
        g = grad_f(x)
        beta = np.dot(g, g - g_prev) / np.dot(d, g_prev)
        d = -g + beta * d_prev

        if verbose:
            print("Iteration {}: x = {}, f(x) = {}".format(i+1, x, f(x)))

        if np.linalg.norm(g) < epsilon:
            if verbose:
                print("Converged in {} iterations".format(i+1))
            break

    if not verbose:
        print("Converged in {} iterations".format(i+1))

    return x, x_hist

# вспомогательная функция для поиска оптимального шага
def line_search_func(alpha, x, d, f, grad_f):
    return f(x + alpha * d)

# вызываем метод сопряженных градиентов с начальной точкой x0 = [0, 0]
x_opt, x_hist = conjugate_gradient_method(np.array([0, 0]), f, grad_f, verbose=True)

# строим график траектории вычислений метода сопряженных градиентов
x_hist = np.array(x_hist)
plt.plot(x_hist[:, 0], x_hist[:, 1], 'o-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Conjugate Gradient Method Trajectory')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_hist[:, 0], x_hist[:, 1], np.arange(len(x_hist)), 'o-')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Iteration')
ax.set_title('Conjugate Gradient Method Trajectory')
plt.show()


print("Optimal point:", x_opt)
print("Optimal value:", f(x_opt))