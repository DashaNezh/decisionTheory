import numpy as np
from scipy.optimize import linprog

# Данные задачи
C = np.array([[2, 4, 3, 2],  # Стоимости перевозки
              [3, 1, 2, 3],
              [5, 4, 1, 5]])

a = np.array([60, 65, 70])  # Запасы
b = np.array([40, 60, 70, 25])  # Потребности

# Количество источников и пунктов назначения
num_sources, num_destinations = C.shape

# Массив для коэффициентов целевой функции (стоимость перевозки)
c = C.flatten()

# Ограничения для запасов (сумма поставок с каждого источника не должна превышать его запас)
A_eq_supply = np.zeros((num_sources, num_sources * num_destinations))
for i in range(num_sources):
    A_eq_supply[i, i * num_destinations:(i + 1) * num_destinations] = 1
b_eq_supply = a

# Ограничения для потребностей (сумма поставок для каждого потребителя должна быть равна потребности)
A_eq_demand = np.zeros((num_destinations, num_sources * num_destinations))
for j in range(num_destinations):
    A_eq_demand[j, j::num_destinations] = 1
b_eq_demand = b

# Составляем все ограничения
A_eq = np.vstack([A_eq_supply, A_eq_demand])
b_eq = np.concatenate([b_eq_supply, b_eq_demand])

# Решаем задачу линейного программирования с помощью linprog
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))

# Выводим результат
if result.success:
    print("Оптимальная стоимость перевозки:", result.fun)
    # Преобразуем решение в матрицу для удобства
    X_optimal = result.x.reshape(num_sources, num_destinations)
    print("Оптимальная матрица перевозок:")
    print(X_optimal)
else:
    print("Не удалось найти решение.")
