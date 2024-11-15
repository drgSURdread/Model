import matplotlib.pyplot as plt
import numpy as np
from analytic_solver import AnalyticSolver
from object_data import ControlObject
from sud_data_class import MotionControlSystem

class LamereyDiagram:
    def __init__(self, channel_name: str, parameter_name: str, value_lst: list):
        self.channel_name = channel_name
        self.results = dict()

        self.k = MotionControlSystem.k[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ]
        self.a = MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ]
        self.g = MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ]
        self.h = MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ]
        self.alpha = MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ]

        # Константы для аналитических уравнений
        self.b = self.a - self.g
        self.d = self.a + self.g
        self.e1 = self.alpha
        self.e2 = self.alpha - self.h
        self.e3 = -self.alpha
        self.e4 = -self.alpha + self.h

        self.__calculate_boundary_points()

        self.y_values = []

    def __calculate_boundary_points(self) -> None:
        self.y_min_bound_point = (
            self.k * (self.a - self.g) - np.sqrt(
                (self.k * self.a + np.sqrt(
                    2 * self.g * (2 * self.alpha - self.h)))**2 - \
                2*self.h*(self.a-self.g)
                )
        )
        self.y_max_bound_point = (
            self.k * (self.a - self.g) + np.sqrt(
            (self.k * self.a + np.sqrt(2 * self.g * (
            2 * self.alpha - self.h)))**2 - 2 * self.h * (self.a - self.g))
        )

    def start(self, y_start: float) -> None:
        while self.__check_end_solution(y_start):
            self.y_values.append(y_start)
            y_start = self.__next_step(y_start)

    def __check_end_solution(self, current_y: float) -> bool:
        if abs(current_y - self.y_values) < 1e-7:
            return True
        return False

    def __next_step(self, current_y: float) -> float:
        self.y_values.append(current_y)
        if self.y_min_bound_point < current_y < self.y_max_bound_point:
            next_y = self.__T1_function(current_y)
        else:
            next_y = self.__T2_function(current_y)
        return next_y
    
    def __T1_function(self, current_y: float) -> float:
        y2 = self.b * self.k - np.sqrt(
            (
                current_y - self.b * self.k)**2 + \
                2 * self.b * (self.e1 - self.e2)
            )
        return -self.g * self.k + np.sqrt(
            (y2 + self.g * self.k)**2 + 2 * self.g * (self.e1 - self.e2)
        )

    def __T2_function(self, current_y: float) -> float:
        y2 = self.b * self.k - np.sqrt(
            (current_y - self.b * self.k)**2 + 2 * self.b * (self.e1 - self.e2)
        )
        y3 = -self.g * self.k - np.sqrt(
            (y2 + self.g * self.k)**2 + 2 * self.g * (self.e3 - self.e2)
        )
        y4 = -self.d * self.k + np.sqrt(
            (y3 + self.d * self.k)**2 + 2 * self.d * (self.e4 - self.e3)
        )
        return -self.g * self.k + np.sqrt(
            (y4 + self.g * self.k)**2 + 2 * self.g * (self.e1 - self.e4)
        )
    
    def plot_diagram(self):
        pass

    def __generate_data_points(self):
        return self.y_values[:len(self.y_values) - 2], self.y_values[1:]
    
    def __generate_plot_data(self, y_1: list, y_2: list):
        plot_line_points_x = [y_1[0]]
        plot_line_points_y = [y_2[0]]
        for i in range(1, len(y_1)):
            # Точка на биссектрисе
            plot_line_points_x.append(y_1[i])
            plot_line_points_y.append(y_1[i])
            # Точка вне биссектрисы
            plot_line_points_x.append(y_1[i])
            plot_line_points_y.append(y_2[i])
        return plot_line_points_x, plot_line_points_y

        
