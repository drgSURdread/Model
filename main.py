import sys

sys.path.append("initialization")

import initialization.initial_data_class as initial
from numerical_solver import NumericalSolver
from sud_data_class import MotionControlSystem
from object_data import ControlObject
import time

def start(ref_file_path: str):
    initial.init_objects(ref_file_path)


if __name__ == "__main__":
    start("initialization/DATA_REF.xlsx")

    sol = NumericalSolver()
    start_time = time.time()
    sol.new_solve(end_time=40, step=0.01)
    # 5 секунд симуляции это примерно минута выполнения
    # на new_solve 5 секунд симуляции это 19 секунд выполнения
    print("Время выполнения", time.time() - start_time)


    sol.plot_phase_portrait("nu")
    sol.plot_phase_portrait("gamma")
    sol.plot_phase_portrait("psi")
    #sol.plot_step_diagram()
    sol.plot_F_function_values()
    sol.plot_m_x()
    sol.plot_m_y()
    sol.plot_m_z()
    sol.plot_show()
