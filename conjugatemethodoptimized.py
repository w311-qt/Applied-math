import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def quadratic_function(x):
    return 3*x[0]**2 + 2*x[1]**2-5*x[0]*x[1]+6*x[0]-7*x[1]+8


def gradient(x):
    return np.array([6*x[0]-5*x[1], 4*x[1]-5*x[0]-7])


def conjugate_gradient_method(x0, f, grad_f, epsilon=1e-5, max_iterations=1000, verbose=True):
    x = np.array(x0)
    d = -grad_f(x)
    g = grad_f(x)
    alpha = minimize_scalar(lambda a: f(x + a * d)).x
    x_trajectory = [x]
    iteration = 0

    while np.linalg.norm(g) > epsilon and iteration < max_iterations:
        d_prev = d
        g_prev = g
        x = x + alpha * d
        x_trajectory.append(x)
        g = grad_f(x)
        beta = np.dot(g, g - g_prev) / np.dot(g_prev, g_prev)
        d = -g + beta * d_prev
        alpha = minimize_scalar(lambda a: f(x + a * d)).x
        iteration += 1

        if verbose:
            print(f"Iteration {iteration}: x = {x}, f(x) = {f(x)}, grad_f(x) = {np.linalg.norm(g)}")

    if iteration == max_iterations:
        print("Maximum number of iterations reached!")
    else:
        print("Optimization succeed!")

    x_opt = x

    return x_opt, x_trajectory, iteration


x0 = [0, 0]
x_opt, x_trajectory, iterations = conjugate_gradient_method(x0, quadratic_function, gradient)

# Plot the optimization trajectory
x_vals = np.linspace(-2, 4, 50)
y_vals = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = quadratic_function([X, Y])
plt.contour(X, Y, Z, levels=10, colors='gray')
plt.plot([x[0] for x in x_trajectory], [x[1] for x in x_trajectory], marker='o')
plt.title("Conjugate Gradient Method Trajectory")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

print(x_trajectory)