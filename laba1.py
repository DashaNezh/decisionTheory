import math
import numpy as np
import matplotlib.pyplot as plt

# Определяем функцию и её производные
def f(x):
    return x - math.log(x)


def f_prime(x):
    return 1 - 1 / x


def f_double_prime(x):
    return 1 / (x ** 2)


# Метод касательных
def tangent_method(a, b, tol=1e-6):
    # Xmin переменная для хранения текущего приближения
    # Вычисляем начальное приближение
    Xmin, c = 0, (f(b) - f(a) + f_prime(a) * a - f_prime(b) * b) / (f_prime(a) - f_prime(b))

    # Счётчик итераций
    iterations = 0

    # Запускаем цикл, пока не достигнем точности
    while (b - a) > tol:
        iterations += 1  # Количество шагов

        # Если f' в точке c уже 0, значит нашли минимум
        if f_prime(c) == 0:
            Xmin = c
            break

        # Смотрим, с какой стороны искать дальше
        elif f_prime(c) > 0:
            b = c
        else:
            a = c

        # Обновляем значение c
        c = (f(b) - f(a) + f_prime(a) * a - f_prime(b) * b) / (f_prime(a) - f_prime(b))
        Xmin = c
    return Xmin, iterations  # Возвращаем минимум и количество шагов

# Метод золотого сечения
def golden_section_search(f, a, b, tol=1e-5):
    phi = (1 + np.sqrt(5)) / 2  # Число золотого сечения
    resphi = 2 - phi  # resphi используется для определения нач. точек

    # Начальные точки
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    # Массив для хранения промежуточных значений
    points = [(x1, f1), (x2, f2)]

    iterations = 0  # Счётчик итераций

    while abs(b - a) > tol:
        iterations += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

        points.append((x1, f1))
        points.append((x2, f2))

    return (a + b) / 2, points, iterations

# Метод Ньютона
def newton_method(x0, tol=1e-6, max_iter=400):
    x = x0
    iterations = 0
    for _ in range(max_iter):
        x_new = x - f_prime(x) / f_double_prime(x)
        iterations += 1
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x, iterations

# Начальные значения
a = 0.1
b = 2.0
x0 = 1.5

# Метод касательных
x_min, iterations_1 = tangent_method(a, b)
y_min = f(x_min)
print("Метод касательных:")
print(f"x = {x_min:.6f}; f(x) = {y_min:.6f}; Итераций: {iterations_1}")

# Метод золотого сечения
minimum, points, iterations_2 = golden_section_search(f, a, b)
print("Метод золотого сечения:")
print(f"x = {minimum:.6f}, f(x) = {f(minimum):.6f}; Итераций: {iterations_2}")

# Метод Ньютона
minimum, iterations_3 = newton_method(x0)
print("Метод Ньютона:")
print(f"x = {minimum:.6f}, f(x) = {f(minimum):.6f}, Итераций: {iterations_3}")

# Построение графиков функции и её производных
x_vals = np.linspace(0.1, 2, 400)
f_vals = [f(x) for x in x_vals]
f_prime_vals = [f_prime(x) for x in x_vals]
f_double_prime_vals = [f_double_prime(x) for x in x_vals]

plt.figure(figsize=(12, 8))

# График функции f(x)
plt.subplot(3, 1, 1)
plt.plot(x_vals, f_vals, label='f(x) = x - ln(x)', color='blue')
plt.plot(x_min, y_min, 'ro', label=f'Точка минимума ({x_min:.3f}, {y_min:.3f})')
plt.title('График функции и её производных')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()

# График f'(x)
plt.subplot(3, 1, 2)
plt.plot(x_vals, f_prime_vals, label="f'(x) = 1 - 1/x", color='green')
plt.ylabel("f'(x)")
plt.grid(True)
plt.legend()

# График f''(x)
plt.subplot(3, 1, 3)
plt.plot(x_vals, f_double_prime_vals, label="f''(x) = 1/x^2", color='red')
plt.xlabel('x')
plt.ylabel("f''(x)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
