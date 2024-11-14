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
        self.results = {np.float64(1.0): {'G2': np.float64(0.005436746905643897)}, np.float64(1.4736842105263157): {'G55': np.float64(0.0016639682994388596)}, np.float64(1.9473684210526314): {'G25': np.float64(0.0009290463603926173)}, np.float64(2.4210526315789473): {'G17': np.float64(0.0006415522167733315)}, np.float64(2.894736842105263): {'G11': np.float64(0.00041717216299175263)}, np.float64(3.3684210526315788): {'G9': np.float64(0.0003409573925152908)}, np.float64(3.8421052631578947): {'G7': np.float64(0.00026585026283619084)}, np.float64(4.315789473684211): {'G5': np.float64(0.00019004370934490048)}, np.float64(4.789473684210526): {'G5': np.float64(0.00018992234735430288)}, np.float64(5.263157894736842): {'G3': np.float64(0.00011364284219335057)}, np.float64(5.7368421052631575): {'G3': np.float64(0.00011406015312004907)}, np.float64(6.2105263157894735): {'G3': np.float64(0.00011372484802487051)}, np.float64(6.684210526315789): {'G1': np.float64(3.787204938213007e-05)}, np.float64(7.157894736842105): {'G1': np.float64(3.8045989166847965e-05)}, np.float64(7.63157894736842): {'G1': np.float64(3.788816668569801e-05)}, np.float64(8.105263157894736): {'G1': np.float64(3.80355483664924e-05)}, np.float64(8.578947368421051): {'G1': np.float64(3.803085368419211e-05)}, np.float64(9.052631578947368): {'G1': np.float64(3.802825098894889e-05)}, np.float64(9.526315789473683): {'G1': np.float64(3.7905147942376344e-05)}, np.float64(10.0): {'G1': np.float64(3.801680976379711e-05)}}
        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray, fast_solve: bool = False) -> None:
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
                    MotionControlSystem.borehole = 0.0
                    MotionControlSystem.count_impulse = 0
                    MotionControlSystem.period = 0.0
                    # Для каждого НУ инициализируем решатель и решаем, пока не получим цикл
                    print("Начинаем движение из точки ({}, {})".format(angle_start * 180 / np.pi, velocity_start * 180 / np.pi))
                    
                    self.__set_zero_lst_to_control_object(
                        nu=(angle_start, velocity_start)
                    )
                    
                    sol = AnalyticSolver(self.channel_name)
                    sol.solve(time_solve=50000.0, dt_max=0.1)
                    if fast_solve and MotionControlSystem.period != 0.0:
                        flag_save = True
                        self.__save_cycles_parameters(param_value, nu=(angle_start, velocity_start))
                        break
                if fast_solve and flag_save:
                    break
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

