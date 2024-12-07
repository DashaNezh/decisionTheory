import numpy as np

# Данные точки
x1 = -0.38057296
x2 = -1.37402052

# Целевая функция
def f_exact(x1, x2):
    return x1**2 + np.exp(x1**2 + x2**2) + 4 * x1 + 3 * x2

# Рассчитать значение функции
f_value = f_exact(x1, x2)
print(f_value)