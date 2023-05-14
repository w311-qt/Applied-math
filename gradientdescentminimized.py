import numpy as np
from matplotlib import pyplot as plt


def gradient_descent(f, x0, eps=1e-6, max_iter=100000, alpha=1e-3):
    """
    Реализация градиентного спуска для минимизации квадратичной функции.
    """
    x = x0.copy()
    for i in range(max_iter):
        grad = f(x)
        if np.linalg.norm(grad) < eps:
            return i
        x -= alpha * grad
    return max_iter
def generate_quadratic(n, k):
    # Генерируем случайную матрицу n на n с числами от -k до k
    A = np.random.uniform(low=-k, high=k, size=(n, n))
    # Делаем матрицу симметричной, чтобы получить квадратичную функцию
    A = np.triu(A) + np.triu(A, 1).T
    # Генерируем случайный вектор n на 1 с числами от -k до k
    b = np.random.uniform(low=-k, high=k, size=(n, 1))
    # Считаем значение квадратичной функции в точке x
    def f(x):
        return 0.5 * x.T @ A @ x + b.T @ x
    return f
ns = [2, 4, 8, 16, 32, 64, 128]  # Размерности пространства
ks = [1, 5, 10, 100]  # Значения числа обусловленности

# Для каждой комбинации n и k генерируем 7 случайных функций и считаем среднее число итераций
for n in ns:
    for k in ks:
        iters = []
        for i in range(7):
            f = generate_quadratic(n, k)
            x0 = np.random.uniform(low=-k, high=k, size=(n, 1))
            iters.append(gradient_descent(f, x0))
        avg_iters = sum(iters) / len(iters)
        print(print(format(f"n = {n}, k = {k}, {avg_iters}\n")))
