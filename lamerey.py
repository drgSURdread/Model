import matplotlib.pyplot as plt
import numpy as np
from analytic_solver import AnalyticSolver
from object_data import ControlObject
from sud_data_class import MotionControlSystem

class LamereyDiagram:
    def __init__(self, channel_name: str):
        self.channel_name = channel_name
        self.results = dict()

        self.k = MotionControlSystem.k[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ][0]
        self.a = MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ][0]
        self.g = MotionControlSystem.g[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ][0]
        self.h = MotionControlSystem.h[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ][0]
        self.alpha = MotionControlSystem.alpha[
            MotionControlSystem.index_channel_mapping[self.channel_name]
        ][0]

        # Константы для аналитических уравнений
        self.b = self.a - self.g
        self.d = self.a + self.g
        self.e1 = self.alpha
        self.e2 = self.alpha - self.h
        self.e3 = -self.alpha
        self.e4 = -self.alpha + self.h

        self.__calculate_boundary_points()

        self.y_values = []
        self.type_function_lst = [] # Хранит типы функций последования применяющиеся на данном шаге

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

    def start(self, y_start: float) -> int:
        self.y_values.append(float(y_start))
        self.type_function_lst.append(None)
        y_start, type_function = self.__next_step(y_start)
        while not(self.__check_end_solution(y_start)):
            self.y_values.append(float(y_start))
            self.type_function_lst.append(type_function)
            y_start, type_function = self.__next_step(y_start)
        
        sum_count_impulse = 0
        for type_func in self.type_function_lst[self.find_index:]:
            sum_count_impulse += int(type_func[1])
        return sum_count_impulse


    def __check_end_solution(self, current_y: float) -> bool:
        if np.any(np.abs(np.array(self.y_values) - current_y) < 1e-7):
            self.find_index = np.where(np.abs(np.array(self.y_values) - current_y) < 1e-7)[0][0]
            return True
        return False

    def __next_step(self, current_y: float) -> tuple:
        if self.y_min_bound_point < current_y < self.y_max_bound_point:
            next_y = self.__T1_function(current_y)
            type_function = "T1"
        else:
            next_y = self.__T2_function(current_y)
            type_function = "T2"
        return next_y, type_function
    
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
    
    def plot_diagram(self, figure_size: tuple = (10, 8)):
        y_1, y_2 = self.__generate_data_points()
        points_x, points_y = self.__generate_plot_data(y_1, y_2)
        borders = self.__get_borders(points_x, points_y)
        ax = self.__get_figure(figure_size)

        rad_to_deg = lambda x: x * 180 / np.pi
        
        ax.axis(
            [
                rad_to_deg(borders["x"]["min"]) - 0.3 * borders["x"]["min"],
                rad_to_deg(borders["x"]["max"]) + 0.1 * borders["x"]["max"],
                rad_to_deg(borders["y"]["min"]) - 0.3 * borders["y"]["min"],
                rad_to_deg(borders["y"]["max"]) + 0.1 * borders["y"]["max"]
            ]
        )

        # Биссектриса
        ax.plot(
            np.linspace(borders["x"]["min"], borders["x"]["max"], 2),
            np.linspace(borders["x"]["min"], borders["x"]["max"], 2),
            color='g'
        )
        
        rad_to_deg = lambda x: x * 180 / np.pi

        # Точки и линии через биссектрису
        ax.scatter(
            list(map(rad_to_deg, y_1)), 
            list(map(rad_to_deg, y_2))
        )
        ax.plot(
            list(map(rad_to_deg, points_x)), 
            list(map(rad_to_deg, points_y))
        )

        plt.title("Диаграмма Ламерея")
        # plt.show()

    def __generate_data_points(self):
        return self.y_values[:len(self.y_values) - 1], self.y_values[1:]
    
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
    
    def __get_figure(self, figure_size: tuple = (10, 8)) -> plt.Axes:
        fig, ax = plt.subplots(figsize=figure_size, layout="tight")
        ax.grid(True)
        plt.xlabel("Y, град./c", fontsize=14, fontweight="bold")
        plt.ylabel("Y', град./c", fontsize=14, fontweight="bold")
        return ax
    
    def __get_borders(self, y_1: list, y_2: list) -> dict:
        return {
            "x":
            {
                "min": min(self.y_values),
                "max": max(self.y_values)
            },
            "y":
            {
                "min": min(self.y_values),
                "max": max(self.y_values)
            }
        }