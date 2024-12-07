import numpy as np

# Определение начальных параметров
c = np.array([1, -1, 0, 0, 0, 0], dtype=float)  # Коэффициенты целевой функции
A = np.array([
    [-1, 1, 1, 0, 0, 0],  # x1 - x2 + s1 = 8
    [8, 5, 0, 1, 0, 0],  # 8x1 + 5x2 + s2 = 80
    [1, -2, 0, 0, 1, 0],  # x1 - 2x2 + s3 = 2
    [-1, -4, 0, 0, 0, 1]  # -x1 - 4x2 + s4 = -4
], dtype=float)  # Приведение к типу float
b = np.array([8, 80, 2, -4], dtype=float)  # Правая часть ограничений

# Добавляем искусственные переменные
A = np.hstack((A, np.eye(A.shape[0])))  # Добавление искусственных переменных
c = np.hstack((c, np.zeros(A.shape[0])))  # Обновление коэффициентов целевой функции


# Симплекс-метод
def simplex(c, A, b):
    m, n = A.shape
    while True:
        # Шаг 1: Определение базисных переменных
        basic_vars = np.where(np.sum(A != 0, axis=0) == 1)[0]

        # Шаг 2: Определение текущего решения
        solution = np.zeros(n)
        for i in basic_vars:
            if np.any(A[:, i] != 0):
                solution[i] = b[np.where(A[:, i] != 0)[0][0]]

        # Шаг 3: Проверка на оптимальность
        if np.all(c >= 0):
            return solution[:2]  # Возвращаем только x1 и x2

        # Шаг 4: Выбор входящей переменной
        entering = np.argmin(c)

        # Шаг 5: Выбор выходящей переменной
        ratios = np.divide(b, A[:, entering], out=np.full_like(b, np.inf), where=A[:, entering] > 0)
        leaving = np.argmin(ratios)

        # Шаг 6: Обновление базиса
        pivot = A[leaving, entering]
        A[leaving, :] /= pivot
        b[leaving] /= pivot

        for i in range(m):
            if i != leaving:
                factor = A[i, entering]
                A[i, :] -= factor * A[leaving, :]
                b[i] -= factor * b[leaving]

        c -= c[entering] * A[leaving, :]


# Решение задачи
solution = simplex(c, A, b)

# Вычисляем максимальное значение целевой функции
max_value = solution[0] - solution[1]  # F = x1 - x2

# Вывод результатов
print("Оптимальное решение:")
print(f"x1 = {solution[0]}, x2 = {solution[1]}")
print(f"Максимальное значение целевой функции: {max_value}")
