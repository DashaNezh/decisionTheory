import numpy as np

# Данные точки
x1 = -1.77054366
x2 = -2.47887563

# Целевая функция
def f_exact(x1, x2):
    return x1**2 + np.exp(x1**2 + x2**2) + 4 * x1 + 3 * x2

# Рассчитать значение функции
f_value = f_exact(x1, x2)
print(f_value)
