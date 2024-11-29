import sys
sys.path.append("initialization")

import matplotlib.pyplot as plt
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
    ) -> EnergyDiagram:    
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

def plot_contour(diagram_obj: EnergyDiagram, type_cycle: str) -> None:
    diagram_obj.plot_contour(type_cycle)

def plot_3d_diagram(diagram_obj: EnergyDiagram, type_cycle: str) -> None:
    diagram_obj.plot_3d_diagram(type_cycle)

def plot_all_surfaces(diagram_obj: EnergyDiagram) -> None:
    diagram_obj.plot_all_surfaces()

if __name__ == "__main__":
    # Пример использования программы:

    # # 1) Инициализируем систему
    # #    При желании потом можно изменить какой-нибудь параметр
    start("initialization/DATA_REF.xlsx")
    
    # # 2) Построение фазового портрета методом точечных отображения
    # # В данном случае не учитываются нелинейности
    sol = analytic_solution(
        channel_name="nu",
        time_solve=100.0
    )
    sol.plot_phase_portrait("nu") # Отобразить фазовый портрет

    # # 3) Анализ переходного процесса методом диаграммы Ламерея
    lamerey_diagram(
        channel_name="nu",
        y_start=0.01,
        beta=0.0001, # Если значение равно 0, то зона нечувствительности не будет учитываться
    )

    # # 4) Построение 2D диаграммы скважности
    # Параметры для построения энергетической диаграммы
    channel_name = "nu"                        # Название канала
    parameter_name_1 = "g"                     # Название варьируемого параметра
    value_lst_1 = np.linspace(1e-7, 2e-7, 100) # Значения варьируемого параметра
    parameter_name_2 = "k"
    value_lst_2 = np.linspace(10, 18, 100)
    NU_matrix = [                        
        np.array([0.0] * 2),
        np.linspace(0.002389*np.pi/180, 0.004389*np.pi/180, 4)
    ] # Набор начальных условий

    energy_diagram(
        channel_name, 
        parameter_name_1, 
        value_lst_1, 
        NU_matrix, 
        P_max=15, 
        P_const=3, 
        beta=0.001389*np.pi/180
    )
    
    # # 5) Построение 3D диаграммы скважности
    diagram_obj = energy_3d_diagram(
        channel_name, 
        parameter_name_1, 
        value_lst_1, 
        parameter_name_2, 
        value_lst_2, 
        NU_matrix,
        beta=0.001389*np.pi/180
    )

    plot_3d_diagram(diagram_obj, type_cycle='Г2')
    plot_all_surfaces(diagram_obj)
    plot_contour(diagram_obj, type_cycle='Г2')

    # # 6) Для валидации с Model можно вывести параметры СУД следующим образом:
    print(MotionControlSystem.a)
    print(MotionControlSystem.g)
    print(MotionControlSystem.alpha * 180 / np.pi)
    print(MotionControlSystem.h * 180 / np.pi)
    print(MotionControlSystem.k)

    plt.show()

    
    
    
