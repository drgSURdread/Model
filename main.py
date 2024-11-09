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
import initialization.initial_data_class as initial
from numerical_solver import NumericalSolver
from analytic_solver import AnalyticSolver
from sud_data_class import MotionControlSystem
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
def analytic_solution(time_solve: float = 10.0):
    """
    Запускает аналитический решатель
    """
    sol = AnalyticSolver("gamma")
    start_time = time.time()

    # HELP: Настройка dt_max позволяет ускорить решатель, но получить более грубое решение.
    # Пока гоняешь модель на разных параметрах, можно ослабить этот параметр. На финальном графике
    # можно убрать настройку, либо выставить 0.01
    sol.solve(time_solve=time_solve, dt_max=0.05)
    print("Время выполнения", time.time() - start_time)
    return sol

if __name__ == "__main__":
    start("initialization/DATA_REF.xlsx")

    sol = analytic_solution(time_solve=20000.0)
    sol.plot_phase_portrait("gamma")
    
    
