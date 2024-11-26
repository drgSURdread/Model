'''
HELP: В процессе моделирования движения с перекрестными связями
на фазовых портретах наблюдаются изломы. Есть идеи, что это из-за
срабатывания реле в других каналах. Впринципе это прдтверждается 
графиками

HELP: Решения с редуцированной и простой матрицей направляющих
косинусов дают одинаковые результаты. А разница во времени решения
около 20 сек

HELP: С текущими настройками решателя наблюдается хорошая точность
около линий перключения. Текущая скорость симуляции - 
100 сек. симуляции == 40 сек. решения
'''

import sys
sys.path.append("initialization")

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import initialization.initial_data_class as initial
from numerical_solver import NumericalSolver
from analytic_solver import AnalyticSolver
from sud_data_class import MotionControlSystem
from object_data import ControlObject
from energy_diagram import EnergyDiagram
from lamerey import LamereyDiagram, NonLinearLamereyDiagram
import time
from calculate_moments import ComputeMoments

def start(ref_file_path: str):
    """
        Запускает инциализацию параметров

    Args:
        ref_file_path (str): путь до `.xlsx` файла
    """
    initial.init_objects(ref_file_path)

def get_moments(channel_name: str):
    """
    Возвращает значения гравитационного момента (для отладки)
    Args:
        channel_name (str): название канала
    """
    index_channel = MotionControlSystem.index_channel_mapping[channel_name]
    disturbing_moment = ComputeMoments.gravitation_moment(reduced=True)
    return disturbing_moment[index_channel, 0]

# TODO: Добавить передачу параметров решателя
def numerical_solution():
    """
    Запускает численный решатель
    """
    sol = NumericalSolver(reduced=True)
    start_time = time.time()
    sol.new_solve(end_time=100)
    print("Время выполнения", time.time() - start_time)
    return sol

# TODO: Добавить передачу параметров решателя
def analytic_solution(channel_name: str = "gamma", time_solve: float = 10.0):
    """
    Запускает аналитический решатель
    """
    sol = AnalyticSolver(channel_name)
    start_time = time.time()

    # HELP: Настройка dt_max позволяет ускорить решатель, но получить более грубое решение.
    # Пока гоняешь модель на разных параметрах, можно ослабить этот параметр. На финальном графике
    # можно убрать настройку, либо выставить 0.01
    sol.solve(time_solve=time_solve, dt_max=0.1)
    print("Время выполнения", time.time() - start_time)
    return sol

def lamerey_diagram(channel_name: str, y_start: float, beta: float = 0.0) -> None:
    if channel_name == "nu":
        ControlObject.nu_angles = [0.0]
        ControlObject.nu_w = [y_start]
    elif channel_name == "psi":
        ControlObject.psi_angles = [0.0]
        ControlObject.psi_w = [y_start]
    else:
        ControlObject.gamma_angles = [0.0]
        ControlObject.gamma_w = [y_start]
    if beta == 0.0:
        diagram = LamereyDiagram(channel_name)
    else:
        diagram = NonLinearLamereyDiagram(channel_name, beta)
    diagram.start(y_start)
    diagram.plot_diagram()

def energy_diagram(
        channel_name: str, 
        parameter_name: str, 
        value_lst: list, 
        NU_matrix: list,
        P_max: float = 0.0,
        P_const: float = 0.0,
        beta: float = 0.0,
    ) -> None:
    MotionControlSystem.set_parameter_value(
        channel_name,
        parameter_name,
        parameter_value=value_lst[0],
    )
    
    diagram = EnergyDiagram(
        channel_name=channel_name,
        parameter_name_1=parameter_name,
        value_lst_1=value_lst,
        P_max=P_max,
        P_const=P_const,
    )
    start_time = time.time()
    diagram.start(nu_matrix=NU_matrix, used_lamerey=True, beta=beta)
    print("Общее время построения диаграммы скважности: ", time.time() - start_time)
    diagram.plot_diagram()

def energy_3d_diagram(
        channel_name: str, 
        parameter_name_1: str, 
        value_lst_1: list, 
        parameter_name_2: str, 
        value_lst_2: list, 
        NU_matrix: list,
        beta: float = 0.0,
    ) -> None:    
    diagram = EnergyDiagram(
        channel_name=channel_name,
        parameter_name_1=parameter_name_1,
        value_lst_1=value_lst_1,
        parameter_name_2=parameter_name_2,
        value_lst_2=value_lst_2,
    )
    start_time = time.time()
    diagram.start(nu_matrix=NU_matrix, used_lamerey=True, beta=beta, diagram_3d=True)
    print("Общее время построения диаграммы скважности: ", time.time() - start_time)
    diagram.plot_3d_diagram()

if __name__ == "__main__":

    start("initialization/DATA_REF.xlsx")
    # Параметры для построения энергетической диаграммы
    channel_name = "nu"                  # Название канала
    parameter_name_1 = "g"                 # Название варьируемого параметра
    value_lst_1 = np.linspace(6e-8, 2e-7, 5)   # Значения варьируемого параметра
    parameter_name_2 = "k"
    value_lst_2 = np.linspace(0.1, 18, 20)
    NU_matrix = [                        # Набор начальных условий
        np.array([0.0] * 2),
        np.linspace(0.002389*np.pi/180, 0.004389*np.pi/180, 4)
    ]
    
    # energy_diagram(
    #     channel_name, 
    #     parameter_name, 
    #     value_lst, 
    #     NU_matrix, 
    #     P_max=15, 
    #     P_const=3, 
    #     beta=0.001389*np.pi/180
    # )

    energy_3d_diagram(
        channel_name, 
        parameter_name_1, 
        value_lst_1, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix,
        beta=0.001389*np.pi/180
    )

    # MotionControlSystem.set_parameter_value(
    #     channel_name,
    #     parameter_name,
    #     parameter_value=12,
    # )
    # lamerey_diagram(channel_name, 0.0025345365*np.pi/180, beta=0.001389*np.pi/180)

    # print(MotionControlSystem.a)
    # print(MotionControlSystem.g)
    # print(MotionControlSystem.alpha * 180 / np.pi)
    # print(MotionControlSystem.h * 180 / np.pi)
    # print(MotionControlSystem.k)
    # print(NU_matrix[1][1] * 180 / np.pi)
    
    # fig = go.Figure(go.Surface(
    # contours = {
    #     "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"white"},
    #     "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
    # },
    # x = [1,2,3,4,5],
    # y = [1,2,3,4,5],
    # z = [
    #     [0, 1, 0, 1, 0],
    #     [1, 0, 1, 0, 1],
    #     [0, 1, 0, 1, 0],
    #     [1, 0, 1, 0, 1],
    #     [0, 1, 0, 1, 0]
    # ]))
    # fig.update_layout(
    #         scene = {
    #             "xaxis": {"nticks": 20},
    #             "zaxis": {"nticks": 4},
    #             'camera_eye': {"x": 0, "y": -1, "z": 0.5},
    #             "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    #         })
    # fig.show()

    # plt.show()

    
    
    
