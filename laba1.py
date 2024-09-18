import numpy as np


# Функция f(x) = x - ln(x)
def f(x):
    return x - np.log(x)


# Первая производная f'(x) = 1 - 1/x
def df(x):
    return 1 - 1 / x


# Вторая производная f''(x) = 1/x^2
def ddf(x):
    return 1 / (x ** 2)


# Метод нулевого порядка: метод золотого сечения
def golden_section_search(a, b, tol=1e-6):
    phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
    resphi = 2 - phi
    iterations = 0

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)

    while abs(b - a) > tol:
        iterations += 1
        if f(c) < f(d):
            b = d
        else:
            a = c

        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

    return (a + b) / 2, iterations


# Метод Ньютона (касательных)
def newton_method(x0, tol=1e-6, max_iter=10000):
    x = x0
    iterations = 0
    for i in range(max_iter):
        iterations += 1
        grad = df(x)
        hessian = ddf(x)
        if abs(grad) < tol:
            break
        if abs(hessian) < tol:  # Избегаем деления на 0
            print("Вторая производная слишком мала, метод может не сойтись.")
            break
        x = x - grad / hessian
    return x, iterations


# Применение методов к функции на интервале [0.1, 2]
a, b = 0.1, 2
tol = 1e-6

# Метод золотого сечения
golden_result, golden_iterations = golden_section_search(a, b, tol)
print(
    f"Метод золотого сечения: минимум в точке x = {golden_result:.6f}, f(x) = {f(golden_result):.6f}, итераций: {golden_iterations}")

# Метод Ньютона (начальная точка 1.5)
initial_x = 1.5
newton_result, newton_iterations = newton_method(initial_x)
print(
    f"Метод Ньютона: минимум в точке x = {newton_result:.6f}, f(x) = {f(newton_result):.6f}, итераций: {newton_iterations}")