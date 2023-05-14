import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad, x0, learning_rate, num_iterations, eps = 1e-7):
    x = np.copy(x0)
    trajectory = np.zeros((num_iterations, 2))
    trajectory[0, :] = x
    iterations = 0
    for i in range(1, num_iterations):
        iterations +=1
        x = x - learning_rate * grad(x)
        trajectory[i, :] = x
        if np.linalg.norm(grad(x)) < eps:
            break
    return x, trajectory, iterations

def plot_trajectory(trajectory, f):
    x_min, x_max = trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1
    y_min, y_max = trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = f([X, Y])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 10))
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-o', color='b', markersize=5, linewidth=3)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Gradient')

    plt.show()


# Example usage with a quadratic function
f = lambda x: x[0] ** 2 + x[1] ** 2 - 2 * x[1] + x[0] ** 2
grad = lambda x: np.array([2 * x[0] + 2 * x[0], 2 * x[1] - 2])

#def f(x):
 #   return 3*x[0]**2 + 4*x[1]**2 - 2*x[0]*x[1] - 3*x[0] - 2*x[1] + 2
#def grad_f(x):
 #   return np.array([6*x[0] - 2*x[1] - 3, 8*x[1] - 2*x[0] - 2])
#def f(x):
 #   return x[0]**2 + x[1]**2
#def grad_f(x):
  #  return np.array([2*x[0], 2*x[1]])
#def f(x):
 #   return x[0] ** 2 + x[1] ** 2 - 2 * x[1] + x[0] ** 2
#def grad_f(x):
 #   return np.array([2 * x[0] + 2 * x[0], 2 * x[1] - 2])

x0 = np.array([1.0, 1.0])
learning_rate = 0.2
num_iterations = 200

x, trajectory, iterations = gradient_descent(f, grad, x0, learning_rate, num_iterations)
plot_trajectory(trajectory, f)
print(f"Оптимальная точка: {x}")
print(f"Итераций: {iterations}")
