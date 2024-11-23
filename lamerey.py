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

    def start(self, y_start: float) -> tuple[int, dict]:
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
        # Иногда дублируются найденные циклы из-за точности
        if len(self.type_function_lst[self.find_index:]) % 2 == 0:
            sum_count_impulse //= 2
        cycle_characteristic = self.__calculate_cycle_characteristics()
        
        return sum_count_impulse, cycle_characteristic
    
    def __calculate_cycle_characteristics(self):
        point_x = self.alpha - self.k * self.y_values[-1]
        self.__set_start_point(
            start_point=(point_x, self.y_values[-1])
        )
        sum_time_impulse = 0.0
        period = 0.0
        count_impulse = 0
        sum_power = 0.0
        sol = AnalyticSolver(self.channel_name, used_lamerey=True)
        first_true = True # TODO: Исправить этот костыль
        while abs(
            ControlObject.get_velocity_value_in_channel(self.channel_name) - self.y_values[-1]
            ) > 1e-7 or first_true:
            first_true = False
            start_time_curve = ControlObject.time_points[-1]
            sol.solve( # Делаем шаг в виде одной кривой
                step_solver=True,
                check_cycle=False,
                dt_max=0.1,
            )
            time_for_curve = ControlObject.time_points[-1] - start_time_curve
            if sol.phase_plane_obj.current_curve == "G0" or sol.phase_plane_obj.current_curve == "G0":
                sum_time_impulse += time_for_curve
                count_impulse += 1
                sum_power += MotionControlSystem.P_max * time_for_curve
            else:
                sum_power += MotionControlSystem.P_const * time_for_curve
            period += time_for_curve

        MotionControlSystem.borehole = sum_time_impulse / period
        MotionControlSystem.period = period
        MotionControlSystem.count_impulse = count_impulse
        MotionControlSystem.power = sum_power / period
        print("Методом диаграммы Ламерея рассчитали следующие параметры ПЦ")
        print("Скважность: ", MotionControlSystem.borehole)
        print("Период: ", MotionControlSystem.period)
        print("Потребляемая мощность: ", MotionControlSystem.power)
        return {
            "borehole": MotionControlSystem.borehole,
            "period": MotionControlSystem.period,
        }

    def __set_start_point(self, start_point: tuple) -> None:
        if self.channel_name == "nu":
            ControlObject.nu_angles = [start_point[0]]
            ControlObject.nu_w = [start_point[1]]
        elif self.channel_name == "psi":
            ControlObject.psi_angles = [start_point[0]]
            ControlObject.psi_w = [start_point[1]]
        else:
            ControlObject.gamma_angles = [start_point[0]]
            ControlObject.gamma_w = [start_point[1]]

    def __check_end_solution(self, current_y: float) -> bool:
        if np.any(np.abs(np.array(self.y_values) - current_y) < 1e-12):
            self.find_index = np.where(np.abs(np.array(self.y_values) - current_y) < 1e-12)[0][-1]
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
            np.linspace(rad_to_deg(borders["x"]["min"]), rad_to_deg(borders["x"]["max"]), 2),
            np.linspace(rad_to_deg(borders["x"]["min"]), rad_to_deg(borders["x"]["max"]), 2),
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


class NonLinearLamereyDiagram(LamereyDiagram):
    def __init__(self, channel_name: str, beta: float) -> object:
        super().__init__(channel_name)
        self.beta = beta

        self.boundary_points = {
            "L1": dict(),
            "L2": dict(),
            "L3": dict(),
            "L4": dict()
        }
        self.__calculate_boundary_points()

    def __calculate_boundary_points(self) -> None:
        self.__calculate_boundary_points_on_line_1()
        self.__calculate_boundary_points_on_line_2()
        self.__calculate_boundary_points_on_line_3()
        self.__calculate_boundary_points_on_line_4()

    def __calculate_boundary_points_on_line_1(self):
        a = self.k * (self.a - self.g)
        b = np.sqrt(
            self.a**2 * self.k**2 + 2 * self.a * np.sqrt(self.g) * self.k * \
            np.sqrt(
                - 2 * self.h + self.g * self.k**2 + 4 * self.alpha - \
                2 * self.k * self.beta
            ) - \
            2 * self.a * (self.h + 2 * self.k * self.beta) + \
            self.g * (self.g * self.k**2 + 4 * self.alpha + 2 * self.k * self.beta)
        )
        # Границы чистого T2 и T1 преобразования
        self.boundary_points['L1']['GR_max'] = a + b 
        self.boundary_points['L1']['GR_min'] = a - b

        b = np.sqrt(
            (self.a - self.g) * (-2 * self.h + \
            (self.a - self.g) * self.k**2) + \
            2 * (self.g - self.a) * self.k * self.beta + \
            self.beta**2
        )
        # Границы скользящего режима на L1
        self.boundary_points['L1']['SK_max'] = a + b 
        self.boundary_points['L1']['SK_min'] = a - b
        
    def __calculate_boundary_points_on_line_2(self):
        a = -self.g * self.k
        b = np.sqrt(
            self.g**2 * self.k**2 + self.beta**2 - \
            2 * self.g * (self.h - 2 * self.alpha + self.k * self.beta)
        )
        # Граница для типа T2 преобразования
        self.boundary_points['L2']['GR_T2'] = a - b

    def __calculate_boundary_points_on_line_3(self):
        a = -self.k * (self.a + self.g)
        b = np.sqrt(
            (self.a + self.g) * (2 * self.h + self.k**2 * (self.a + self.g)) - \
            2 * (self.a + self.g) * self.k * self.beta + self.beta**2
        )
        # Граница для скользящего режима на L3
        self.boundary_points['L3']['SK_max'] = a + b 
        self.boundary_points['L3']['SK_min'] = a - b

    def __calculate_boundary_points_on_line_4(self):
        a = -self.g * self.k
        b = np.sqrt(
            self.g**2 * self.k**2 + self.beta**2 + 2 * self.g * (
                self.h - self.k * self.beta
            )
        )
        # Граница T2 преобразования на L4
        self.boundary_points['L4']['GR_T2'] = a - b

    def start(self, y_start: float) -> tuple[int, dict]:
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
        # Иногда дублируются найденные циклы из-за точности
        if len(self.type_function_lst[self.find_index:]) % 2 == 0:
            sum_count_impulse //= 2
        cycle_characteristic = self.__calculate_cycle_characteristics()
        
        return sum_count_impulse, cycle_characteristic
    
    def __next_step(self, current_y: float) -> tuple:
        if self.y_min_bound_point < current_y < self.y_max_bound_point:
            next_y = self.__T1_function(current_y)
            type_function = "T1"
        else:
            next_y = self.__T2_function(current_y)
            type_function = "T2"
        return next_y, type_function