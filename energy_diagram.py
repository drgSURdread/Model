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

        # self.results = dict()
        self.results = {np.float64(10.0): {'G1': np.float64(3.8061766950372053e-05), 'G2': np.float64(3.795672744777635e-05)}, np.float64(11.052631578947368): {'G1': np.float64(3.7882157958372294e-05)}, np.float64(12.105263157894736): {'G1': np.float64(3.8034645640086427e-05)}, np.float64(13.157894736842106): {'G1': np.float64(3.790193250956863e-05)}, np.float64(14.210526315789473): {'G1': np.float64(3.7918370032905784e-05)}, np.float64(15.263157894736842): {'G1': np.float64(3.801575162906061e-05)}, np.float64(16.315789473684212): {'G1': np.float64(3.80083686569093e-05)}, np.float64(17.36842105263158): {'G1': np.float64(3.8003448196500165e-05)}, np.float64(18.421052631578945): {'G1': np.float64(3.799737117039478e-05)}, np.float64(19.473684210526315): {'G1': np.float64(3.7940371264199504e-05)}, np.float64(20.526315789473685): {'G1': np.float64(3.794586119116573e-05)}, np.float64(21.57894736842105): {'G1': np.float64(3.797672103597023e-05)}, np.float64(22.63157894736842): {'G1': np.float64(3.794447264806094e-05)}, np.float64(23.684210526315788): {'G1': np.float64(3.7951774109486456e-05)}, np.float64(24.736842105263158): {'G1': np.float64(3.7983064326410665e-05)}, np.float64(25.789473684210527): {'G1': np.float64(3.797317662957775e-05)}, np.float64(26.842105263157894): {'G1': np.float64(3.795537244089037e-05)}, np.float64(27.894736842105264): {'G1': np.float64(3.795677222432783e-05)}, np.float64(28.94736842105263): {'G1': np.float64(3.7952701998180464e-05)}, np.float64(30.0): {'G1': np.float64(3.796741728087348e-05)}}
        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray) -> None:
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
            print(
                "Для {} = {}".format(self.parameter_name, param_value), 
                self.results
            )

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
                "type_cycle": "G{}".format(MotionControlSystem.count_impulse),
                "borehole": MotionControlSystem.borehole,
            }
        }

    def __save_results(self) -> None:
        # FIXME: Переписать это уродство
        for _ in self.cycles.keys(): # Перебираем НУ
            for param_value in self.cycles[_].keys(): # Перебираем данные полученных циклов
                if param_value not in self.results:
                    self.results[param_value] = dict()
                if self.cycles[_][param_value]["type_cycle"] not in self.results[param_value]:
                    self.results[param_value][self.cycles[_][param_value]["type_cycle"]] = \
                        self.cycles[_][param_value]["borehole"]
                
    def __generate_plot_data(self) -> dict:
        dict_data = dict()
        for param_value in self.results: # Проходим по всем значениям варьируемого параметра
            cycles = self.results[param_value]
            for type_cycle in cycles.keys(): # Проходим по всем найденным циклам
                if type_cycle not in dict_data:
                    dict_data[type_cycle] = []
                dict_data[type_cycle].append((param_value, cycles[type_cycle]))
        plot_data = dict()
        for type_cycle in dict_data.keys():
            plot_data[type_cycle] = [[], []]
            for point in dict_data[type_cycle]:
                plot_data[type_cycle][0].append(point[0])
                plot_data[type_cycle][1].append(point[1])
        return plot_data

    def plot_diagram(self, figure_size: tuple = (10, 8)) -> None:
        ax = self.__get_figure()
        #borders = self.__set_borders(self.channel_name, ax)
        plot_data = self.__generate_plot_data()
        plt.xlabel(self.parameter_name, fontsize=14, fontweight="bold")
        plt.ylabel("λ", fontsize=14, fontweight="bold")
        for type_cycle in plot_data.keys():
            ax.plot(
                plot_data[type_cycle][0],
                plot_data[type_cycle][1],
                label=type_cycle,
                linewidth=3,
            )
        ax.legend(fontsize=14)
        plt.show()

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
        # ax.grid(which="major", color="#DDDDDD", linewidth=1.5)
        # ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=1)
        # ax.minorticks_on()
        ax.grid(True)
        return ax

