import matplotlib.pyplot as plt
import numpy as np
from analytic_solver import AnalyticSolver
from object_data import ControlObject
from sud_data_class import MotionControlSystem

class EnergyDiagram:
    def __init__(self, channel_name: str, parameter_name: str, value_lst: list):
        self.channel_name = channel_name
        self.parameter_name = parameter_name
        self.value_lst = value_lst

        self.results = dict()
        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray) -> None:
        """
        Перебираем НУ в квадрате со сторонами [angle_min, angle_max] и [velocity_min, velocity_max]
        """
        # Уже предвкушаю, как я не дождусь
        for param_value in self.value_lst:
            self.cycles = dict()
            for angle_start in nu_matrix[0]:
                for velocity_start in nu_matrix[1]:
                    MotionControlSystem.borehole = 0.0
                    MotionControlSystem.count_impulse = 0
                    MotionControlSystem.period = 0.0
                    # Для каждого НУ инициализируем решатель и решаем, пока не получим цикл
                    print("Начинаем движение из точки ({}, {})".format(angle_start * 180 / np.pi, velocity_start * 180 / np.pi))
                    
                    self.__set_zero_lst_to_control_object(
                        nu=(angle_start, velocity_start)
                    )
                    
                    sol = AnalyticSolver(self.channel_name)
                    sol.solve(time_solve=40000.0, dt_max=0.1)
                    
                    self.__save_cycles_parameters(param_value, nu=(angle_start, velocity_start))
            self.__save_results()

    def __set_zero_lst_to_control_object(self, nu: tuple) -> None:
        if self.channel_name == "nu":
            ControlObject.nu_angles = [nu[0]]
            ControlObject.nu_w = [nu[1]]
        elif self.channel_name == "psi":
            ControlObject.psi_angles = [nu[0]]
            ControlObject.psi_w = [nu[1]]
        else:
            ControlObject.gamma_angles = [nu[0]]
            ControlObject.gamma_w = [nu[1]]

    def __save_cycles_parameters(self, parameter_value: float, nu: tuple) -> None:
        self.cycles[nu] = {
            parameter_value: {
                "type_cycle": "G{}".format(MotionControlSystem.count_impulse),
                "borehole": MotionControlSystem.borehole,
            }
        }

    def __save_results(self) -> None:
        for _ in self.cycles.keys(): # Перебираем НУ
            self.results[self.cycles[_]] = dict()
            for param_value in self.cycles[_]: # Перебираем данные полученных циклов
                if self.cycles[_][param_value]["type_cycle"] not in self.results[param_value]:
                    self.results[param_value][self.cycles[_][param_value]["type_cycle"]] = \
                        self.cycles[_][param_value]["borehole"]
                


