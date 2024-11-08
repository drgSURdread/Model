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
    initial.init_objects(ref_file_path)

def get_moments(channel_name: str):
    index_channel = MotionControlSystem.index_channel_mapping[channel_name]
    disturbing_moment = ComputeMoments.gravitation_moment(reduced=True)
    return disturbing_moment[index_channel, 0]

def numerical_solution():
    sol = NumericalSolver(reduced=True)
    start_time = time.time()
    sol.new_solve(end_time=100)
    print("Время выполнения", time.time() - start_time)
    return sol

def analytic_solution():
    sol = AnalyticSolver("nu")
    start_time = time.time()
    sol.solve()
    print("Время выполнения", time.time() - start_time)
    return sol

if __name__ == "__main__":
    start("initialization/DATA_REF.xlsx")

    sol = analytic_solution()
    sol.plot_phase_portrait("nu")
    
    
