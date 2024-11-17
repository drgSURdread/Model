import matplotlib.pyplot as plt
import numpy as np
from analytic_solver import AnalyticSolver
from object_data import ControlObject
from sud_data_class import MotionControlSystem
from lamerey import LamereyDiagram

class EnergyDiagram:
    def __init__(self, channel_name: str, parameter_name: str, value_lst: list):
        self.channel_name = channel_name
        self.parameter_name = parameter_name
        self.value_lst = value_lst

        self.results = dict()
        # self.results = {np.float64(1.0): {'G2': np.float64(0.005436746905643897)}, np.float64(1.11864406779661): {'G2': np.float64(0.004212003091287311)}, np.float64(1.2372881355932204): {'G2': np.float64(0.003135126736940051)}, np.float64(1.3559322033898304): {'G2': np.float64(0.0042225364021006625)}, np.float64(1.4745762711864407): {'G55': np.float64(0.0016647381034759672)}, np.float64(1.5932203389830508): {'G43': np.float64(0.001453785726508056)}, np.float64(1.711864406779661): {'G35': np.float64(0.0012477780709302642)}, np.float64(1.8305084745762712): {'G29': np.float64(0.00106318888733145)}, np.float64(1.9491525423728815): {'G25': np.float64(0.0009290939585969581)}, np.float64(2.0677966101694913): {'G23': np.float64(0.0008602379585332042)}, np.float64(2.1864406779661016): {'G21': np.float64(0.0007884278863960911)}, np.float64(2.305084745762712): {'G19': np.float64(0.0007160707928644217)}, np.float64(2.423728813559322): {'G17': np.float64(0.0006421065924327584)}, np.float64(2.5423728813559325): {'G15': np.float64(0.0005675181339542384)}, np.float64(2.6610169491525424): {'G13': np.float64(0.0004924714611956155)}, np.float64(2.7796610169491527): {'G13': np.float64(0.0004919279591496196)}, np.float64(2.898305084745763): {'G11': np.float64(0.00041718474759898594)}, np.float64(3.016949152542373): {'G11': np.float64(0.0004172012348819831)}, np.float64(3.135593220338983): {'G9': np.float64(0.0003416009246839555)}, np.float64(3.2542372881355934): {'G9': np.float64(0.00034098398412759094)}, np.float64(3.3728813559322033): {'G9': np.float64(0.0003409511500061331)}, np.float64(3.4915254237288136): {'G7': np.float64(0.0002652739585326862)}, np.float64(3.610169491525424): {'G7': np.float64(0.00026524058131075504)}, np.float64(3.728813559322034): {'G7': np.float64(0.00026524343155633626)}, np.float64(3.847457627118644): {'G7': np.float64(0.0002658487288393474)}, np.float64(3.9661016949152543): {'G5': np.float64(0.00019006288965357258)}, np.float64(4.084745762711865): {'G5': np.float64(0.00019007552230578076)}, np.float64(4.203389830508474): {'G5': np.float64(0.0001899417109386194)}, np.float64(4.322033898305085): {'G5': np.float64(0.00018945986392658563)}, np.float64(4.4406779661016955): {'G5': np.float64(0.00018946255619497354)}, np.float64(4.559322033898305): {'G5': np.float64(0.0001899936526601776)}, np.float64(4.677966101694915): {'G5': np.float64(0.00018950017124738883)}, np.float64(4.796610169491526): {'G5': np.float64(0.00018953143580675337)}, np.float64(4.915254237288136): {'G3': np.float64(0.00011362422926523215)}, np.float64(5.033898305084746): {'G3': np.float64(0.00011363173002123979)}, np.float64(5.152542372881356): {'G3': np.float64(0.00011364619908342912)}, np.float64(5.271186440677966): {'G3': np.float64(0.00011365394695555076)}, np.float64(5.389830508474576): {'G3': np.float64(0.00011408269731644995)}, np.float64(5.508474576271187): {'G3': np.float64(0.00011366183027352053)}, np.float64(5.627118644067797): {'G3': np.float64(0.00011406138598594074)}, np.float64(5.745762711864407): {'G3': np.float64(0.00011405947573148787)}, np.float64(5.864406779661017): {'G3': np.float64(0.00011404088173015409)}, np.float64(5.983050847457627): {'G3': np.float64(0.00011371348771622404)}, np.float64(6.101694915254237): {'G3': np.float64(0.00011371799904636472)}, np.float64(6.220338983050848): {'G3': np.float64(0.00011372844825743382)}, np.float64(6.338983050847458): {'G3': np.float64(0.00011374417370298814)}, np.float64(6.457627118644068): {'G3': np.float64(0.00011398585891933143)}, np.float64(6.576271186440678): {'G3': np.float64(0.00011376173211197717)}, np.float64(6.694915254237288): {'G1': np.float64(3.78740765871604e-05)}, np.float64(6.813559322033899): {'G1': np.float64(3.805210089599161e-05)}, np.float64(6.932203389830509): {'G1': np.float64(3.78815633369672e-05)}, np.float64(7.0508474576271185): {'G1': np.float64(3.788001118631977e-05)}, np.float64(7.169491525423729): {'G1': np.float64(3.78797391762235e-05)}, np.float64(7.288135593220339): {'G1': np.float64(3.788227203698991e-05)}, np.float64(7.406779661016949): {'G1': np.float64(3.804641959110239e-05)}, np.float64(7.52542372881356): {'G1': np.float64(3.804043707364645e-05)}, np.float64(7.6440677966101696): {'G1': np.float64(3.788943431407816e-05)}, np.float64(7.762711864406779): {'G1': np.float64(3.803788372598148e-05)}, np.float64(7.88135593220339): {'G1': np.float64(3.789168083610398e-05)}, np.float64(8.0): {'G1': np.float64(3.803397012653025e-05)}}
        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray, fast_solve: bool = False) -> None:
        pass

    def __solution_used_lamerey(self, nu_matrix: list) -> None:
        for param_value in self.value_lst:
            self.cycles = dict()
            MotionControlSystem.set_parameter_value(
                self.channel_name,
                self.parameter_name,
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
                diagram = LamereyDiagram(self.channel_name)
                count_impulse, cycle_parameters = diagram.start(velocity_start)
                self.__save_cycles_parameters(
                    parameter_value=param_value,
                    nu=(0.0, velocity_start)
                )
            self.__save_results()

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
                    if MotionControlSystem.period != 0.0:
                        flag_save = True
                        self.__save_cycles_parameters(param_value, nu=(angle_start, velocity_start))
                        break
                if flag_save:
                    break
            self.__save_results()
            # print(
            #     "Для {} = {}".format(self.parameter_name, param_value), 
            #     self.results
            # )

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
        # plt.show()

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

