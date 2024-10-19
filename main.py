import sys

sys.path.append("initialization")
import matplotlib.pyplot as plt

import initialization.initial_data_class as initial
from numerical_solver import NumericalSolver
from sud_data_class import MotionControlSystem
from object_data import ControlObject
import time
from calculate_moments import ComputeMoments
import numpy as np

def start(ref_file_path: str):
    initial.init_objects(ref_file_path)

def get_moments(channel_name: str):
    index_channel = MotionControlSystem.index_channel_mapping[channel_name]
    disturbing_moment = ComputeMoments.gravitation_moment(reduced=True)
    return disturbing_moment[index_channel, 0]

def solution():
    sol = NumericalSolver(reduced=True)
    start_time = time.time()
    sol.new_solve(end_time=50)
    print("Время выполнения", time.time() - start_time)
    return sol

# HELP: В процессе моделирования движения с перекрестными связями
# на фазовых портретах наблюдаются изломы. Есть идеи, что это из-за
# срабатывания реле в других каналах. Впринципе это прдтверждается 
# графиками
# HELP: Решения с редуцированной и простой матрицей направляющих
# косинусов дают одинаковые результаты. А разница во времени решения
# около 20 сек.

# HELP: С текущими настройками решателя наблюдается хорошая точность
# около линий перключения. Текущая скорость симуляции - 
# 100 сек. симуляции == 40 сек. решения
if __name__ == "__main__":
    start("initialization/DATA_REF.xlsx")

    sol = solution()

    sol.plot_phase_portrait("nu")
    sol.plot_phase_portrait("gamma")
    sol.plot_phase_portrait("psi")

    #sol.plot_F_function_values()
    sol.plot_step_diagram()

    #sol.plot_disturbing_moment_gamma()
    #sol.plot_disturbing_moment_nu()
    #sol.plot_disturbing_moment_psi()

    #sol.plot_disturbing_moment_gamma(angles=True)
    #sol.plot_disturbing_moment_nu(angles=True)
    #sol.plot_disturbing_moment_psi(angles=True)

    #sol.plot_oscillogram_gamma()
    #sol.plot_oscillogram_psi()
    #sol.plot_oscillogram_nu()

    sol.plot_show()
    
    
