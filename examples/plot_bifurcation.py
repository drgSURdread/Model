# %%
import sys
sys.path.append("/Users/evgenijkondratev/Desktop/Udis/Model")
from main import *
# %% [markdown]
# Инициализируем систему управления
start("../initialization/DATA_REF.xlsx")
# %%
import matplotlib.pyplot as plt
import pandas as pd
# %% [markdown]
# Считывание найденных бифуркационных значений в Wolfram
df = pd.read_csv('test.csv')
val_lst = []
for i, val in enumerate(df.columns):
    if i == 0:
        value = float(val[1:])
    elif i == df.columns.shape[0] - 1:
        value = float(val[:-1])
    else:
        value = float(val)
    val_lst.append(value)
val_lst
# %% [markdown]
# Параметры для построения энергетической диаграммы
channel_name = "nu"                        # Название канала
parameter_name_1 = "g"                     # Название варьируемого параметра
value_lst_1 = np.linspace(4e-10, 2.5e-8, 30) # Значения варьируемого параметра
NU_matrix = [                        
    np.array([0.0] * 2),
    np.linspace(0.002389*np.pi/180, 0.004389*np.pi/180, 4)
] # Набор начальных условий

MotionControlSystem.set_parameter_value(
    channel_name,
    "k",
    20.1825100000,
)
MotionControlSystem.set_parameter_value(
    channel_name,
    "alpha",
    0.1008027600 * np.pi / 180,
)
MotionControlSystem.set_parameter_value(
    channel_name,
    "h",
    0.0201607200 * np.pi / 180,
)
MotionControlSystem.set_parameter_value(
    channel_name,
    "a",
    0.0035070500,
)
# %%
diagram = EnergyDiagram(
        channel_name=channel_name,
        parameter_name_1=parameter_name_1,
        value_lst_1=value_lst_1,
        P_max=0,
        P_const=0,
    )
# Запускаем на простом разбиении
diagram.start(nu_matrix=NU_matrix, used_lamerey=True, beta=0)
# Считаем скважности на найденных бифуркационных значениях
diagram.value_lst_1 = [4.54904e-9]
diagram.plot_bif_values = True
diagram.start(nu_matrix=NU_matrix, used_lamerey=True, beta=0)
diagram.plot_diagram()
# %%
if channel_name == "nu":
        ControlObject.nu_angles = [0.0]
        ControlObject.nu_w = [NU_matrix[1][0]]
elif channel_name == "psi":
    ControlObject.psi_angles = [0.0]
    ControlObject.psi_w = [NU_matrix[1][0]]
else:
    ControlObject.gamma_angles = [0.0]
    ControlObject.gamma_w = [NU_matrix[1][0]]
MotionControlSystem.set_parameter_value(
    channel_name,
    "g",
    0.995e-8,
)
lamerey = LamereyDiagram(channel_name)
lamerey.start(0.001*np.pi/180)
lamerey.plot_diagram()
# %%
if channel_name == "nu":
        ControlObject.nu_angles = [0.0]
        ControlObject.nu_w = [NU_matrix[1][0]]
elif channel_name == "psi":
    ControlObject.psi_angles = [0.0]
    ControlObject.psi_w = [NU_matrix[1][0]]
else:
    ControlObject.gamma_angles = [0.0]
    ControlObject.gamma_w = [NU_matrix[1][0]]
MotionControlSystem.set_parameter_value(
    channel_name,
    "g",
    0.9e-8,
)
lamerey = LamereyDiagram(channel_name)
lamerey.start(0.00067*np.pi/180)
lamerey.plot_diagram()

# %%
0.9e-8 * 20.1825100000 * 180/np.pi
# %%
