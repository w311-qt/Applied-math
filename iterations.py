import numpy as np
import matplotlib.pyplot as plt

#def quadratic(x):
 #   return 3*x[0]**2 + 4*x[1]**2 - 2*x[0]*x[1] - 3*x[0] - 2*x[1] + 2
#def grad_quadratic(x):
 #   return np.array([6*x[0] - 2*x[1] - 3, 8*x[1] - 2*x[0] - 2])
#def quadratic(x):
 #   return x[0]**2 + x[1]**2
#def grad_quadratic(x):
 #   return np.array([2*x[0], 2*x[1]])
def quadratic(x):
    return x[0] ** 2 + x[1] ** 2 - 2 * x[1] + x[0] ** 2
def grad_quadratic(x):
    return np.array([2 * x[0] + 2 * x[0], 2 * x[1] - 2])


# Алгоритм сходящегося градиента
def gradient_descent(x_init, lr, tol, max_iter):
    x = x_init
    iter_num = 0
    x_list = []
    x_list.append(x_init)
    while iter_num < max_iter:
        grad = grad_quadratic(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        iter_num += 1
        x_new=list(x)
        x_list.append(x_new)
        print("Iteration {}: x = {}, f(x) = {}".format(iter_num, x, quadratic(x)))
    return x, quadratic(x), iter_num, np.array(x_list)

# Задаем начальное значение, шаг градиентного спуска, точность и максимальное число итераций
x_init = np.array([1.0, 1.0]) # изменен тип на float64
# lr = 0.12
# tol = 1e-8
# max_iter = 300

lr = 0.31
tol = 1e-7
max_iter = 100

# Выполняем алгоритм градиентного спуска
x_star, f_x_star, num_iters, x_list = gradient_descent(x_init, lr, tol, max_iter)
x_list = x_list[1:]

# Выводим результаты
print("Optimal value= ", x_star)
print("Iterations: ", num_iters)

# # Рисуем график итераций
plt.plot(x_list[:, 0], x_list[:, 1], 'o-', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent')
plt.show()

x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_grid = quadratic([x_grid, y_grid])

fig, ax = plt.subplots()
ax.contourf(x_grid, y_grid, z_grid, cmap='gray', alpha=0.3)
ax.plot([p[0] for p in x_list], [p[1] for p in x_list], marker='o', markersize=1, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()
#
# # Построение графика итераций
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = quadratic([X, Y])
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='gray', alpha=0.3)
plt.plot(*zip(*x_list), '-o')
plt.title("Gradient")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


delta = 0.025
x = np.arange(-0.4, 0.6, delta)
y = np.arange(-0.5, 0.5, delta)
X, Y = np.meshgrid(x, y)
Z = quadratic([X, Y])
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Gradient')

# plot steepest descent iterations
x_list = np.array(x_list)
plt.plot(x_list[:, 0], x_list[:, 1], 'bo-')
plt.show()
print(x_list)

def f(x):
    return x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2

#
print(f([32, 32])-f([31,31]))
print(f([31,31])-f([30,30]))
print(f([32, 32])-f([31,31]) >= f([31,31])-f([30,30]))
#
def g(x):
    return x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 6*x[0] - 8*x[1] + 13
print(g([32, 32])-g([31,31]))
print(g([31,31])-g([30,30]))
print(g([32, 32])-g([31,31]) >= g([31,31])-g([30,30]))

def d(x):
    return x[0]**2 + 2 * x[1]**2
print(d([32, 32])-d([31,31]))
print(d([31,31])-d([30,30]))
print(d([32, 32])-d([31,31]) >= d([31,31])-d([30,30]))