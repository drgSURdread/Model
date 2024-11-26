import matplotlib.pyplot as plt
import numpy as np
from analytic_solver import AnalyticSolver
from object_data import ControlObject
from sud_data_class import MotionControlSystem
from lamerey import LamereyDiagram, NonLinearLamereyDiagram
import plotly.graph_objects as go

class EnergyDiagram:
    # TODO: Это не класс, а туториал по тому как не надо писать код
    def __init__(
            self, 
            channel_name: str, 
            parameter_name_1: str, 
            value_lst_1: list,
            parameter_name_2: str = None,
            value_lst_2: list = None,
            P_max: float = 0.0,
            P_const: float = 0.0):
        self.channel_name = channel_name
        self.parameter_name_1 = parameter_name_1
        self.value_lst_1 = value_lst_1
        self.parameter_name_2 = parameter_name_2
        self.value_lst_2 = value_lst_2

        MotionControlSystem.P_max = P_max
        MotionControlSystem.P_const = P_const

        self.results = dict()
        # self.results_x = dict()
        # self.results_y = dict()
        self.plot_data = dict()
        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray, used_lamerey: bool = False, beta: float = 0.0, diagram_3d: bool = False) -> None:
        if used_lamerey:
            if diagram_3d:
                self.__solution_used_3d_lamerey(nu_matrix, beta)
            else:
                self.__solution_used_lamerey(nu_matrix, beta)
        else:
            self.__iterate_solution(nu_matrix)

    def __solution_used_lamerey(self, nu_matrix: list, beta: float = 0.0) -> None:
        for param_value in self.value_lst_1:
            self.cycles = dict()
            MotionControlSystem.set_parameter_value(
                self.channel_name,
                self.parameter_name_1,
                param_value,
            )
            for velocity_start in nu_matrix[1]:
                MotionControlSystem.borehole = 0.0
                MotionControlSystem.count_impulse = 0
                MotionControlSystem.period = 0.0
                print("Начинаем движение из точки ({}, {})".format(0.0, velocity_start * 180 / np.pi))
                self.__set_zero_lst_to_control_object(
                        nu=(0.0, velocity_start)
                )
                if beta != 0.0:
                    diagram = NonLinearLamereyDiagram(self.channel_name, beta)
                else:
                    diagram = LamereyDiagram(self.channel_name)
                diagram.start(velocity_start)
                self.__save_cycles_parameters(
                    parameter_value=param_value,
                    nu=(0.0, velocity_start)
                )
            self.__save_results()

    def __solution_used_3d_lamerey(self, nu_matrix: list, beta: float = 0.0) -> None:
        # TODO: Повторение в коде. Поправить
        for param_value_1 in self.value_lst_1:
            for param_value_2 in self.value_lst_2:
                self.cycles = dict()
                MotionControlSystem.set_parameter_value(
                    self.channel_name,
                    self.parameter_name_1,
                    param_value_1,
                )
                MotionControlSystem.set_parameter_value(
                    self.channel_name,
                    self.parameter_name_2,
                    param_value_2,
                )
                for velocity_start in nu_matrix[1]:
                    MotionControlSystem.borehole = 0.0
                    MotionControlSystem.count_impulse = 0
                    MotionControlSystem.period = 0.0
                    print("Начинаем движение из точки ({}, {})".format(0.0, velocity_start * 180 / np.pi))
                    self.__set_zero_lst_to_control_object(
                            nu=(0.0, velocity_start)
                    )
                    if beta != 0.0:
                        diagram = NonLinearLamereyDiagram(self.channel_name, beta)
                    else:
                        diagram = LamereyDiagram(self.channel_name)
                    diagram.start(velocity_start)
                    self.__save_3d_cycles_parameters(
                        parameter_value_1=param_value_1,
                        parameter_value_2=param_value_2,
                        nu=(0.0, velocity_start)
                    )
            # self.__save_3d_results()

    def __iterate_solution(self, nu_matrix: list) -> None:
        """
        Перебираем НУ в квадрате со сторонами [angle_min, angle_max] и [velocity_min, velocity_max]
        """
        for param_value in self.value_lst:
            self.cycles = dict()
            MotionControlSystem.set_parameter_value(
                self.channel_name,
                self.parameter_name,
                param_value,
            )
            flag_save = False
            for angle_start in nu_matrix[0]:
                for velocity_start in nu_matrix[1]:
                    # MotionControlSystem.borehole = 0.0
                    # MotionControlSystem.count_impulse = 0
                    # MotionControlSystem.period = 0.0
                    # Для каждого НУ инициализируем решатель и решаем, пока не получим цикл
                    print("Начинаем движение из точки ({}, {})".format(angle_start * 180 / np.pi, velocity_start * 180 / np.pi))
                    
                    self.__set_zero_lst_to_control_object(
                        nu=(angle_start, velocity_start)
                    )
                    
                    sol = AnalyticSolver(self.channel_name)
                    sol.solve(time_solve=50000.0, dt_max=0.1)
                    if MotionControlSystem.period != 0.0:
                        flag_save = True
                        self.__save_cycles_parameters(param_value, nu=(angle_start, velocity_start))
                        break
                if flag_save:
                    break
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
        ControlObject.y_L1 = []
        ControlObject.time_points = [0.0]

    def __save_cycles_parameters(self, parameter_value: float, nu: tuple) -> None:
        self.cycles[nu] = {
            parameter_value: {
                "type_cycle": "Г{}".format(MotionControlSystem.count_impulse),
                "borehole": MotionControlSystem.borehole,
                "power": MotionControlSystem.power,
            }
        }

    def __save_3d_cycles_parameters(self, parameter_value_1: float, parameter_value_2: float, nu: tuple) -> None:
        # self.cycles[nu] = {
        #     (parameter_value_1, parameter_value_2): {
        #         "type_cycle": "Г{}".format(MotionControlSystem.count_impulse),
        #         "borehole": MotionControlSystem.borehole,
        #         "power": MotionControlSystem.power,
        #     }
        # }
        if "Г{}".format(MotionControlSystem.count_impulse) not in self.plot_data.keys():
            self.plot_data["Г{}".format(MotionControlSystem.count_impulse)] = [[], [], []]
        x_array = self.plot_data["Г{}".format(MotionControlSystem.count_impulse)][0]
        y_array = self.plot_data["Г{}".format(MotionControlSystem.count_impulse)][1]
        z_matrix = self.plot_data["Г{}".format(MotionControlSystem.count_impulse)][2]

        if parameter_value_1 not in x_array:
            x_array.append(parameter_value_1)
            # Добавляем новую строку в матрицу для Z(x, y)
            z_matrix.append([])
        if parameter_value_2 not in y_array:
            y_array.append(parameter_value_2)
        z_matrix[x_array.index(parameter_value_1)].append(MotionControlSystem.borehole)

    def __save_results(self) -> None:
        # FIXME: Переписать это уродство
        for _ in self.cycles.keys(): # Перебираем НУ
            for param_value in self.cycles[_].keys(): # Перебираем данные полученных циклов
                if param_value not in self.results:
                    self.results[param_value] = dict()
                if self.cycles[_][param_value]["type_cycle"] not in self.results[param_value]:
                    self.results[param_value][self.cycles[_][param_value]["type_cycle"]] = \
                        [self.cycles[_][param_value]["borehole"], self.cycles[_][param_value]["power"]]
                    
    def __generate_plot_data(self) -> tuple[dict, dict]:
        dict_data = dict()
        power_data = dict()
        for param_value in self.results: # Проходим по всем значениям варьируемого параметра
            cycles = self.results[param_value]
            for type_cycle in cycles.keys(): # Проходим по всем найденным циклам
                if type_cycle not in dict_data:
                    dict_data[type_cycle] = []
                    power_data[type_cycle] = []
                dict_data[type_cycle].append((param_value, cycles[type_cycle][0])) # Сохраняем скважность
                power_data[type_cycle].append((param_value, cycles[type_cycle][1])) # Сохраняем мощность

        plot_data = dict()
        plot_power_data = dict()
        for type_cycle in dict_data.keys():
            plot_data[type_cycle] = [[], []]
            plot_power_data[type_cycle] = [[], []]
            for point in dict_data[type_cycle]:
                plot_data[type_cycle][0].append(point[0])
                plot_data[type_cycle][1].append(point[1])
            for point in power_data[type_cycle]:
                plot_power_data[type_cycle][0].append(point[0])
                plot_power_data[type_cycle][1].append(point[1])
        return plot_data, plot_power_data

    def plot_diagram(self, figure_size: tuple = (10, 8)) -> None:
        plot_data, plot_power_data = self.__generate_plot_data()
        ax = self.__get_figure()
        plt.xlabel(self.parameter_name, fontsize=14, fontweight="bold")
        plt.ylabel("λ", fontsize=14, fontweight="bold")
        for type_cycle in plot_data.keys():
            if len(plot_data[type_cycle][0]) == 1:
                ax.scatter(
                    plot_data[type_cycle][0],
                    plot_data[type_cycle][1],
                    label=type_cycle,
                    s=20,
                )
            else:
                ax.plot(
                    plot_data[type_cycle][0],
                    plot_data[type_cycle][1],
                    label=type_cycle,
                    linewidth=3,
                )
        ax.legend(fontsize=14)

        # ax = self.__get_figure()
        # plt.xlabel(self.parameter_name, fontsize=14, fontweight="bold")
        # plt.ylabel("Мощность, Вт", fontsize=14, fontweight="bold")
        # for type_cycle in plot_power_data.keys():
        #     if len(plot_power_data[type_cycle][0]) == 1:
        #         ax.scatter(
        #             plot_power_data[type_cycle][0],
        #             plot_power_data[type_cycle][1],
        #             label=type_cycle,
        #             s=20,
        #         )
        #     else:
        #         ax.plot(
        #             plot_power_data[type_cycle][0],
        #             plot_power_data[type_cycle][1],
        #             label=type_cycle,
        #             linewidth=3,
        #         )
        # ax.legend(fontsize=14)
        # plt.show()
    
    def plot_3d_diagram(self) -> None:
        # plot_data, plot_power_data = self.__generate_plot_data()
        # ax = self.__get_figure()
        # plt.xlabel(self.parameter_name, fontsize=14, fontweight="bold")
        # plt.ylabel("λ", fontsize=14, fontweight="bold")
        data = []
        for type_cycle in self.plot_data.keys():
            if len(self.plot_data[type_cycle][0]) == 1:
                continue # TODO: Пока пропускаем, потом будем отмечать точки на диаграмме
            else:
                data.append(
                    go.Surface(
                        x=self.plot_data[type_cycle][0], 
                        y=self.plot_data[type_cycle][1], 
                        z=self.plot_data[type_cycle][2]
                    )
                )
        # ax.legend(fontsize=14)
        fig = go.Figure(data=data)
        fig.update_traces(contours_z=dict( 
            show=True, usecolormap=True, 
            highlightcolor="limegreen", 
            project_z=True)
        ) 
        fig.show() 

    def __get_figure(self, figure_size: tuple = (10, 8)) -> plt.Axes:
        """
        Получает фигуру для отображения графика

        Parameters
        ----------
        figure_size : tuple, optional
            Размер фигуры (высота, ширина), by default (10, 8)

        Returns
        -------
        plt.Axes
            Объект фигуры
        """
        fig, ax = plt.subplots(figsize=figure_size, layout="tight")
        ax.grid(True)
        return ax

