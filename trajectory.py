import numpy as np
from scipy.optimize import brent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# определяем квадратичную функцию
def f(x):
    return x[0]**2 + 2 * x[1]**2

# определяем градиент функции
def grad_f(x):
    return np.array([2 * x[0], 4 * x[1]])

# реализуем метод сопряженных градиентов
def conjugate_gradient_method(x0, f, grad_f, epsilon=1e-5, max_iterations=1000, verbose=True):
    x = x0
    d = -grad_f(x)
    g = grad_f(x)
    alpha = 0.1
    x_hist = [x]

    for i in range(max_iterations):
        d_prev = d
        alpha = brent(line_search_func, brack=(0, 1), args=(x, d, f, grad_f))
        x = x + alpha * d
        x_hist.append(x)
        g_prev = g
        g = grad_f(x)
        beta = np.dot(g, g - g_prev) / np.dot(d, g_prev)
        d = -g + beta * d_prev

        if verbose:
            print("Iteration {}: x = {}, f(x) = {}, grad_f(x) = {}".format(i+1, x, f(x), np.linalg.norm(g)))

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# строим поверхность функции
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f([X1, X2])
surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)

# строим траекторию метода
ax.plot(x_hist[:, 0], x_hist[:, 1], f(x_hist.T), 'o-', color='black', linewidth=2)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.set_title('Conjugate Gradient Method Trajectory')

plt.show()
fig, ax = plt.subplots()

# строим поверхность функции
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f([X1, X2])
surf = ax.contourf(X1, X2, Y, cmap='gray', alpha=0.31)

# строим траекторию метода
ax.plot(x_hist[:, 0], x_hist[:, 1], 'o-', color='r', linewidth=2)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Conjugate Gradient Method Trajectory')

# добавляем цветовую шкалу
plt.show()

x1 = np.linspace(-2, 4, 50)
x2 = np.linspace(-4, 4, 50)
X1, X2 = np.meshgrid(x1, x2)
Y = f([X1, X2])
plt.contour(X1, X2, Y, levels=10, colors='gray')
plt.plot([x[0] for x in x_hist], [x[1] for x in x_hist], marker='o')
plt.title("Conjugate Gradient Method Trajectory")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


print("Optimal point:", x_opt)
print("Optimal value:", f(x_opt))