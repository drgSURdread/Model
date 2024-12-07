# %%
import sys
sys.path.append("/Users/evgenijkondratev/Desktop/Udis/Model")
from main import *
# %% [markdown]
# Инициализируем систему управления
start("../initialization/DATA_REF.xlsx")
# %%
import matplotlib.pyplot as plt
# %%
deg_to_rad = lambda x : x * np.pi / 180
rad_to_deg = lambda x : x * 180 / np.pi
# %% [markdown]
# Выведем параметры системы управления
a = 0.0035070500
g = 0.0000000234
alpha = 0.1008027600 * np.pi / 180
h = 0.0201607200 * np.pi / 180
k = 20.1825100000
# %% [markdown]
# # Функции граничных точек на линии L1
# %%
g_values = np.linspace(1e-8, 3e-8, 100)
y_gr_1_values = (
            k * (a - g_values) - np.sqrt(
                (k * a + np.sqrt(
                    2 * g_values * (2 * alpha - h)))**2 - \
                2*h*(a-g_values)
                )
        )
y_gr_2_values = (
            k * (a - g_values) + np.sqrt(
                (k * a + np.sqrt(
                    2 * g_values * (2 * alpha - h)))**2 - \
                2*h*(a-g_values)
                )
        )
# %%
fig, ax = plt.subplots(figsize=(10, 8), layout="tight")
ax.grid(True)
ax.plot(g_values, y_gr_1_values * 180 / np.pi)
# %%
fig, ax = plt.subplots(figsize=(10, 8), layout="tight")
ax.grid(True)
ax.plot(g_values, y_gr_2_values * 180 / np.pi)
# %%
fig, ax = plt.subplots(figsize=(10, 8), layout="tight")
ax.grid(True)
ax.plot(g_values, (y_gr_2_values - y_gr_1_values) * 180 / np.pi)


# %% [markdown]
# # Проверка повторного прохождения через граничную точку для бифуркации Г1 -> Г3
# %%
g = 0.00000002265912
# %%
b = a - g
d = a + g
e1 = alpha
e2 = alpha - h
e3 = -alpha
e4 = -alpha + h
# %%
y3_k = - g * k # Точка касания
y4 = -d * k + np.sqrt(
            (y3_k + d * k)**2 + 2 * d * (e4 - e3)
        )
rad_to_deg(y4)
# %%
y1 = -g * k + np.sqrt(
            (y4 + g * k)**2 + 2 * g * (e1 - e4)
        )
rad_to_deg(y1)
# %%
y2 = b * k - np.sqrt(
            (
                y1 - b * k)**2 + \
                2 * b * (e1 - e2)
            )
rad_to_deg(y2)
# %%
y1 = -g * k + np.sqrt(
            (y2 + g * k)**2 + 2 * g * (e1 - e2)
        )
rad_to_deg(y1)
# %%
y2 = b * k - np.sqrt(
            (
                y1 - b * k)**2 + \
                2 * b * (e1 - e2)
            )
rad_to_deg(y2)
# %%
y3 = -g * k - np.sqrt(
            y2**2 + (g * k)**2 + 2 * g * (h + k * y2 - 2 * alpha)
        )
print(rad_to_deg(y3))
print(y3 - y3_k)
# %%
rad_to_deg(y3_k)


# %% [markdown]
# # Проверка повторного прохождения через граничную точку для бифуркации Г3 -> Г5
# %%
g = 0.0000000098
# %%
b = a - g
d = a + g
e1 = alpha
e2 = alpha - h
e3 = -alpha
e4 = -alpha + h
# %%
y3_k = - g * k # Точка касания
y4 = -d * k + np.sqrt(
            (y3_k + d * k)**2 + 2 * d * (e4 - e3)
        )
rad_to_deg(y4)
# %%
y1 = -g * k + np.sqrt(
            (y4 + g * k)**2 + 2 * g * (e1 - e4)
        )
rad_to_deg(y1)
# %%
y2 = b * k - np.sqrt(
            (
                y1 - b * k)**2 + \
                2 * b * (e1 - e2)
            )
rad_to_deg(y2)
# %%
y1 = -g * k + np.sqrt(
            (y2 + g * k)**2 + 2 * g * (e1 - e2)
        )
rad_to_deg(y1)
# %%
y2 = b * k - np.sqrt(
            (
                y1 - b * k)**2 + \
                2 * b * (e1 - e2)
            )
rad_to_deg(y2)
# %% [markdown]
# Из-за того что порядок бифуркации выше, необходимо применить еще одно 
# дополнительное T2 преобразование
# %%
y3 = -g * k - np.sqrt(
            y2**2 + (g * k)**2 + 2 * g * (h + k * y2 - 2 * alpha)
        )
print(rad_to_deg(y3))
# %%
y4 = -d * k + np.sqrt(
            (y3 + d * k)**2 + 2 * d * (e4 - e3)
        )
rad_to_deg(y4)
# %%
y1 = -g * k + np.sqrt(
            (y4 + g * k)**2 + 2 * g * (e1 - e4)
        )
rad_to_deg(y1)
# %%
# %% [markdown]
# Теперь применяем частичную T2 функцию и проверяем попадание в исходную точку касания
# %%
y2 = b * k - np.sqrt(
            (
                y1 - b * k)**2 + \
                2 * b * (e1 - e2)
            )
rad_to_deg(y2)
# %%
y3 = -g * k - np.sqrt(
            y2**2 + (g * k)**2 + 2 * g * (h + k * y2 - 2 * alpha)
        )
print(rad_to_deg(y3))
print(y3 - y3_k)
# %%
rad_to_deg(y3_k)