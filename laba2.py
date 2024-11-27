import numpy as np
import matplotlib.pyplot as plt


# функция для вычисления значения целевой функции
def f(x1, x2):
    return x1 ** 2 + np.exp(x1 ** 2 + x2 ** 2) + 4 * x1 + 3 * x2


# функция для вычисления градиента целевой функции
def gradient(x):
    x1, x2 = x
    # производные по x1 и x2
    df_dx1 = 2 * x1 + 2 * x1 * np.exp(x1 ** 2 + x2 ** 2) + 4
    df_dx2 = 2 * x2 * np.exp(x1 ** 2 + x2 ** 2) + 3
    return np.array([df_dx1, df_dx2])


# метод нелдера-мида для минимизации функции
def nelder_mead(f, x0, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-6, max_iter=1000):
    n = len(x0)  # размерность пространства
    # создаем начальный симплекс
    simplex = np.array([x0, x0 + np.array([alpha, 0]), x0 + np.array([0, alpha])])
    values = np.array([f(*simplex[i]) for i in range(n + 1)])  # значения функции в вершинах симплекса

    for iter in range(max_iter):
        order = values.argsort()  # сортируем вершины по значению функции
        simplex = simplex[order]
        values = values[order]

        # вычисляем центроид без худшей вершины
        centroid = np.mean(simplex[:-1], axis=0)

        # отражение худшей вершины
        reflected = centroid + alpha * (centroid - simplex[-1])
        reflected_values = f(*reflected)

        # проверка, попадает ли отраженная точка в диапазон улучшений
        if values[0] <= reflected_values < values[-2]:
            simplex[-1] = reflected
            values[-1] = reflected_values
            continue

        # проверка на возможность расширения
        if reflected_values < values[0]:
            expanded = centroid + gamma * (centroid - simplex[-1])
            expanded_values = f(*expanded)

            # заменяем худшую точку на расширенную, если это дает улучшение
            if expanded_values < reflected_values:
                simplex[-1] = expanded
                values[-1] = expanded_values
            else:
                simplex[-1] = reflected
                values[-1] = reflected_values
            continue

        # пробуем сжатие
        contracted = centroid + beta * (simplex[-1] - centroid)
        contracted_value = f(*contracted)

        if contracted_value < values[-1]:
            simplex[-1] = contracted
            values[-1] = contracted_value
        else:
            # уменьшаем размер симплекса
            simplex = simplex[0] + (simplex - simplex[0]) * beta
            values = np.array([f(*simplex[i]) for i in range(n + 1)])

        # проверяем условие сходимости
        if np.max(np.abs(values - values[0])) < tol:
            break

    return simplex[0], values[0], iter + 1  # возвращаем точку минимума, значение функции и число итераций


# градиентный метод с постоянным шагом и уменьшением шага при необходимости
def gradient_descent(f, x0, initial_step=0.1, tol=1e-6, max_iter=1000):
    x = x0
    step_size = initial_step

    for iteration in range(max_iter):
        grad = gradient(x)  # вычисляем градиент
        x_new = x - step_size * grad  # обновляем значение x

        # проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            break

        # проверяем, улучшилось ли значение функции
        if f(*x_new) < f(*x):
            x = x_new  # обновляем x, если значение функции уменьшилось
        else:
            step_size *= 0.5  # уменьшаем шаг, если улучшения нет

    return x, f(*x), iteration + 1  # возвращаем точку минимума, значение функции и число итераций


# начальная точка
x0 = np.array([1, 1])

# запуск метода нелдера-мида
min_point_nelder, min_value_nelder, iterations_nelder = nelder_mead(f, x0)

# запуск градиентного метода
min_point_gradient, min_value_gradient, iterations_gradient = gradient_descent(f, x0)

# вывод результатов метода нелдера-мида
print("Метод нелдера-мида\n")
print(f"Координаты точки минимума: {min_point_nelder}")
print(f"Значение функции в этой точке: {min_value_nelder}")
print(f"Количество итераций: {iterations_nelder}\n")

# вывод результатов градиентного метода
print("Градиентный метод с постоянным шагом\n")
print(f"Координаты точки минимума: {min_point_gradient}")
print(f"Значение функции в этой точке: {min_value_gradient}")
print(f"Количество итераций: {iterations_gradient}")

# визуализация функции и точек минимума
x1 = np.linspace(-2, 4, 400)
x2 = np.linspace(-2, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

plt.figure(figsize=(10, 6))
contour = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)  # добавляем цветовую шкалу
plt.plot(min_point_nelder[0], min_point_nelder[1], 'ro')  # точка минимума методом нелдера-мида
plt.title("Контурная карта функции")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()  # включаем сетку
plt.show()
