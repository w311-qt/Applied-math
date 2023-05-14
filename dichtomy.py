import numpy as np
import matplotlib.pyplot as plt


def quadratic(x):
    return x[0]**2 + 2 * x[1]**2


def gradient_quadratic(x):
    return np.array([2 * x[0], 4 * x[1]])


def dichotomy(f, x, grad, eps=1e-6):
    a = -10.0
    b = 10.0
    delta = eps / 2
    while abs(b - a) > eps:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        if f(x - x1 * grad) < f(x - x2 * grad):
            b = x2
        else:
            a = x1
    return (a + b) / 2


x_start = np.array([6, 6])
alpha = 0.1
x = x_start
path = [x_start]
iterations =0
for i in range(100):
    iterations+=1
    eps = 1e-6
    grad = gradient_quadratic(x)
    alpha_opt = dichotomy(quadratic, x, grad, eps)
    if np.linalg.norm(gradient_quadratic(x)) < eps:
        break
    x = x - alpha_opt * grad
    path.append(x)
    print("Iteration {}: x = {}, f(x) = {}, grad_f(x) = {}".format(i + 1, x, quadratic(x), np.linalg.norm(grad)))

x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
z_grid = quadratic([x_grid, y_grid])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='gray', alpha=0.3)
ax.plot([p[0] for p in path], [p[1] for p in path], [quadratic(p) for p in path], '-o', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

fig, ax = plt.subplots()
ax.contourf(x_grid, y_grid, z_grid, cmap='gray', alpha=0.3)
ax.plot([p[0] for p in path], [p[1] for p in path], marker='o', markersize=4, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()

print("Optimal point:", x)
print(f"Iterations: {iterations}")
