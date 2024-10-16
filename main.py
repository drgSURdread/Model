import sys

sys.path.append("dz/phase_portrait/src/initialization")

import initialization.initial_data_class as initial
from analytic_solver import AnalyticSolver


def start(ref_file_path: str):
    initial.init_objects(ref_file_path)


if __name__ == "__main__":
    start("dz/phase_portrait/src/initialization/DATA_REF.xlsx")
    sol = AnalyticSolver(control_channel="nu")
    sol.solve(count_steps=15)
    sol.plot_phase_portrait(channel_name="nu")
