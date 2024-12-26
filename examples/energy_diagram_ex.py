# %%
import sys
sys.path.append("/Users/evgenijkondratev/Desktop/УДИС КП/Udis/Model")
from main import *
# %%
# Инициализируем систему управления
start("../initialization/DATA_REF.xlsx")
# %%
deg_to_rad = lambda x : x * np.pi / 180
# %%
# Выведем параметры системы управления
print("a = ", *MotionControlSystem.a)
print("g = ", *MotionControlSystem.g)
print("alpha = ", *MotionControlSystem.alpha * 180 / np.pi)
print("h = ", *MotionControlSystem.h * 180 / np.pi)
print("k = ", *MotionControlSystem.k)
# %%
# Параметры для построения энергетической диаграммы
channel_name = "nu"                        # Название канала
parameter_name_1 = "g"                     # Название варьируемого параметра
value_lst_1 = np.linspace(1e-7, 2e-7, 100) # Значения варьируемого параметра
parameter_name_2 = "k"
value_lst_2 = np.linspace(2.5, 18, 500)
NU_matrix = [                        
    np.array([0.0] * 2),
    np.linspace(0.002389*np.pi/180, 0.004389*np.pi/180, 4)
] # Набор начальных условий
# %% [markdown]
# # Построение энергетической диаграммы без учета нелинейностей
# %%
parameter_name_2 = "k"
value_lst_2 = np.linspace(2.5, 18, 700)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.0
    )
# %%
parameter_name_2 = "g"
value_lst_2 = np.linspace(0.4e-8, 0.75e-7, 1000)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.0
    )
# %% [markdown]
# # Свойство масштабной инвариатности
# %%
parameter_name_2 = "k"
value_lst_2 = np.linspace(2.3, 3.5, 100)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.0
    )
# %%
parameter_name_2 = "g"
value_lst_2 = np.linspace(3.5e-8, 9e-8, 500)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.0
    )


# %% [markdown]
# # Построение энергетической диаграммы с нелинейностями

# %%
# Параметры для построения энергетической диаграммы
channel_name = "nu"                        # Название канала
parameter_name_1 = "g"                     # Название варьируемого параметра
value_lst_1 = np.linspace(1e-7, 2e-7, 100) # Значения варьируемого параметра
parameter_name_2 = "k"
value_lst_2 = np.linspace(2.5, 18, 500)
NU_matrix = [                        
    np.array([0.0] * 2),
    np.linspace(0.002389*np.pi/180, 0.004389*np.pi/180, 4)
] # Набор начальных условий
# %%
parameter_name_2 = "k"
value_lst_2 = np.linspace(6, 45, 1000)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.001389*np.pi/180
    )
# %% [markdown]
# Исследуем природу скачка
# %%
MotionControlSystem.set_parameter_value(
    "nu",
    "k",
    30.0,
)
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.003),
    beta=deg_to_rad(0.001389),
)
plt.show()
# %%
# %%
MotionControlSystem.set_parameter_value(
    "nu",
    "k",
    35.0,
)
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.003),
    beta=deg_to_rad(0.001389),
)

# %%
parameter_name_2 = "g"
value_lst_2 = np.linspace(0.4e-9, 2e-7, 2000)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.001389*np.pi/180
    )

# %% [markdown]
# Исследуем природу скачка
# %%
MotionControlSystem.set_parameter_value(
    "nu",
    "g",
    1.0e-7,
)
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.003),
    beta=deg_to_rad(0.001389),
)
plt.show()
# %%
# %%
MotionControlSystem.set_parameter_value(
    "nu",
    "g",
    1.2e-7,
)
lamerey_diagram(
    channel_name="nu",
    y_start=deg_to_rad(0.003),
    beta=deg_to_rad(0.001389),
)


# %% [markdown]
# # Оценка влияния зоны нечувствительности на характер бифуркаций
# %%
parameter_name_2 = "k"
value_lst_2 = np.linspace(3, 20, 2000)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.001629*np.pi/180
    )
# %%
parameter_name_2 = "g"
value_lst_2 = np.linspace(0.4e-9, 2e-7, 2000)
energy_diagram(
        channel_name, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.001029*np.pi/180
    )
# %%
