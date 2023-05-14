import numpy as np
import matplotlib.pyplot as plt

# Определение функции и её градиента
def f(x):
    return x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 6*x[0] - 8*x[1] + 13
def grad_f(x):
    return np.array([2*x[0] + 2*x[1] - 6, 4*x[1] + 2*x[0] - 8])

# Задание начальной точки и параметров алгоритма
x0 = np.array([3, 4])
alpha0 = 0.1
c1 = 0.5
epsilon = 1e-6

# Инициализация списков для хранения значений x и alpha
x_list = [x0]
alpha_list = [alpha0]
iterations =0
# Реализация алгоритма спуска с дроблением шага и условием Армихо
while True:
    iterations += 1
    xk = x_list[-1]
    grad_fk = grad_f(xk)
    alpha_k = alpha_list[-1]
    xk1 = xk - alpha_k * grad_fk
    fk1 = f(xk1)
    fk = f(xk)
    armijo_cond = fk1 <= fk - c1 * alpha_k * np.dot(grad_fk, grad_fk)
    if armijo_cond:
        x_list.append(xk1)
        alpha_list.append(alpha_k)
        if np.linalg.norm(grad_f(xk1)) < epsilon:
            break
    else:
        alpha_list.append(alpha_k / 2)

# Построение графика итераций
x = np.linspace(0, 4, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.contour(X, Y, Z, levels=np.logspace(0.5, 5, 35), cmap='gray', alpha=0.5)
plt.plot(*zip(*x_list), '-o', color='b')
plt.title("Gradient")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

print(f"Optimal point: {x_list[-1]}")
print(f"Iterations: {iterations}")
