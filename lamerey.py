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
        pass
    
    def __next_step(self):
        pass