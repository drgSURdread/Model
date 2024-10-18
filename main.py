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
    sol = NumericalSolver()
    start_time = time.time()
    sol.new_solve(end_time=20, step=0.01)
    # 5 секунд симуляции это примерно минута выполнения
    # на new_solve 5 секунд симуляции это 19 секунд выполнения
    print("Время выполнения", time.time() - start_time)
    return sol

if __name__ == "__main__":
    start("initialization/DATA_REF.xlsx")

    
    sol = solution()
    sol.plot_phase_portrait("nu")
    sol.plot_phase_portrait("gamma")
    sol.plot_phase_portrait("psi")
    #sol.plot_step_diagram()
    #sol.plot_F_function_values()
    sol.plot_m_x()
    #sol.plot_m_y()
    #sol.plot_m_z()
    sol.plot_osc_x()
    sol.plot_osc_y()
    sol.plot_osc_z()

    """
    moments = []
    angles = ControlObject.gamma_angles
    for angle in angles:
        ControlObject.set_angles_in_channel("gamma", angle)
        moments.append(get_moments("gamma"))
    fig, ax = plt.subplots(figsize=(10, 10), layout="tight")
    ax.grid(which="major", color="#DDDDDD", linewidth=1.5)
    ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=1)
    ax.minorticks_on()
    ax.grid(True)
    plt.xlabel("t, c", fontsize=14, fontweight="bold")
    plt.ylabel("M_x, Н * м", fontsize=14, fontweight="bold")
    ax.plot(
        angles,
        moments,
        color="g",
    )
    plt.title("gamma")
    """

    sol.plot_show()
    
    
