# %%
import sys
sys.path.append("/Users/evgenijkondratev/Desktop/Udis/Model")
from main import *
# %%
# Инициализируем систему управления
start("../initialization/DATA_REF.xlsx")
# %%
# Выведем параметры системы управления
print("a = ", *MotionControlSystem.a)
print("g = ", *MotionControlSystem.g)
print("alpha = ", *MotionControlSystem.alpha * 180 / np.pi)
print("h = ", *MotionControlSystem.h * 180 / np.pi)
print("k = ", *MotionControlSystem.k)
# %%
deg_to_rad = lambda x : x * np.pi / 180
# %%
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.01),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
# Построим ближе предельный цикл
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.003),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
MotionControlSystem.set_parameter_value(
    "nu",
    "k",
    10.18251,
)
# %%
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.006),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
# Построим ближе предельный цикл
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.005),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
# Покажем большое количество граничных точек
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(10),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
# Покажем большое количество граничных точек
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(-0.000002),
    beta=deg_to_rad(0.001389), # Если значение равно 0, то зона нечувствительности не будет учитываться
)
# %%
