import math
import numpy as np
import matplotlib.pyplot as plt

# Task 1 and Task 2

def f(x):
    return np.sin(x)

def g(x):
    return x**3 + np.sin(x)

def f_prime(x):
    return np.cos(x)

# вычисление численной производной на узлах сетки
def num_deriv(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

# Определяем аналитические производные
def df(x):
    return np.cos(x)

def dg(x):
    return 3*x**2 - 4*x + 1

def right_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def left_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


x = 1
h = 0.1  # задаем шаг
a = 0
b = np.pi/2
n = 1000
# Определяем сетку
setka = np.linspace(-5, 5, 100)

# Строим графики функций и их аналитических производных
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(setka, f(setka), label='f(x)')
plt.plot(setka, df(setka), label="f'(x)")
plt.legend()
plt.title("График функции f(x)")

plt.subplot(1, 2, 2)
plt.plot(setka, g(setka), label='g(x)')
plt.plot(setka, dg(setka), label="g'(x)")
plt.legend()
plt.title("График функции g(x)")

plt.show()

x0 = 1
df_num = (f(x0 + h) - f(x0 - h)) / (2 * h)
dg_num = (g(x0 + h) - g(x0 - h)) / (2 * h)

# Task3 and Task4

# задаем диапазон значений x
x_range = (0, 1)

# вычисляем численные значения производной и точные значения производной
x_values = []
numerical_derivatives = []
exact_derivatives = []
for x in [i*h for i in range(int((x_range[1]-x_range[0])/h)+1)]:
    x_values.append(x)
    numerical_derivatives.append((f(x+h)-f(x))/h)
    exact_derivatives.append(f_prime(x))

# вычисляем среднеквадратичное отклонение
k = len(x_values)
sum_of_squares = sum([(numerical_derivatives[i]-exact_derivatives[i])**2 for i in range(k)])
sigma = math.sqrt(1/k * sum_of_squares)

# вычисление среднеквадратичного отклонения численной производной от истинной производной
def rmsd(f, f_prime, x, h):
    num = num_deriv(f, x, h)
    true = f_prime(x)
    return np.sqrt(np.mean((num - true)**2))

# вычисление СКО для разных значений шага
h_values = [0.1, 0.05, 0.025, 0.0125]
rmsd_values = [rmsd(f, f_prime, np.pi/4, h) for h in h_values]
plt.loglog(h_values, rmsd_values, 'o-')
plt.xlabel('Step size')
plt.ylabel('RMSD')
plt.title('Convergence of numerical derivative')
plt.show()

# Task5 - Task7

# методы численного интегрирования
def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * np.sum(f(x[:-1]))

def trapezoid_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    return h * (np.sum(f(x[:-1])) + np.sum(f(x[1:])))/2

def simpson_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, 2*n+1)
    return h/3 * np.sum(f(x[0:-2:2]) + 4*f(x[1::2]) + f(x[2::2]))

# compute the analytical values
f_int = 1
g_int = 1/4 + (np.cos(1) - np.cos(0))/2

# compute the numerical values
f_rect = rectangle_rule(f, a, b, n)
f_trap = trapezoid_rule(f, a, b, n)
f_simp = simpson_rule(f, a, b, n)

g_rect = rectangle_rule(g, 0, 1, n)
g_trap = trapezoid_rule(g, 0, 1, n)
g_simp = simpson_rule(g, 0, 1, n)

# Вычисление аналитической производной sin(x) на интервале [0, 2π]
web = np.linspace(0, 2 * np.pi, 100)
y_true = df(web)

# Вычисление численной производной sin(x) на интервале [0, 2π] для разных значений шага h
h_values = [0.1, 0.05, 0.025, 0.0125]
rmsd_val = []
for h in h_values:
    y_numerical = central_difference(f, web, h)
    rmsd1 = np.sqrt(np.mean((y_numerical - y_true)**2))
    rmsd_val.append(rmsd1)

# Построение графика зависимости отклонения от величины шага
plt.plot(h_values, rmsd_val, 'o-')
plt.xscale('log')
plt.xlabel('Шаг h')
plt.ylabel('Отклонение от аналитического решения')
plt.title('Зависимость отклонения от величины шага')
plt.show()
# print the results
print("================Task 1=================")
print("f'(1) по правой разности =", right_diff(f, x, h))
print("f'(1) по левой разности =", left_diff(f, x, h))
print("f'(1) по центральной разности =", central_difference(f,x,h))
print("================Task 2=================")
print("Численная производная функции f(x) в точке x=1:", df_num)
print("Численная производная функции g(x) в точке x=1:", dg_num)
print("================Task 3=================")
print("Среднеквадратичное отклонение: ", sigma)
print("================Task 6=================")
print("---Definite integrals---:")
print("Analytical f(x) integral:", f_int)
print("Rectangle rule f(x) integral:", f_rect)
print("Trapezoid rule f(x) integral:", f_trap)
print("Simpson's rule f(x) integral:", f_simp)
print("Analytical g(x) integral:", g_int)
print("Rectangle rule g(x) integral:", g_rect)
print("Trapezoid rule g(x) integral:", g_trap)
print("Simpson's rule g(x) integral:", g_simp)

print("---Errors compared to analytical values---:")
print("Rectangle rule f(x) error:", np.abs(f_rect - f_int))
print("Trapezoid rule f(x) error:", np.abs(f_trap - f_int))
print("Simpson's rule f(x) error:", np.abs(f_simp - f_int))
print("Rectangle rule g(x) error:", np.abs(g_rect - g_int))
print("Trapezoid rule g(x) error:", np.abs(g_trap - g_int))
print("Simpson's rule g(x) error:", np.abs(g_simp - g_int))