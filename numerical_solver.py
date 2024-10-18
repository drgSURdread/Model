import numpy as np
from object_data import ControlObject
from scipy.integrate import odeint
from scipy import integrate, signal
from numba import njit # TODO: для ускорения, пока без нее
from sud_data_class import MotionControlSystem
import matplotlib.pyplot as plt
from phase_plane import PhasePlane
from calculate_moments import ComputeMoments

# TODO: Улучшить шаги решателя
# Сейчас долго решает. Надо увеличивать шаг на G0

# TODO: Улучшить код визуализации

class NumericalSolver:
    def __init__(self, reduced: bool = False) -> None:
        self.__reduced_cosine_matrix = reduced

        # Отладочная информация
        self.step_size_lst = []

        self.gamma_relay_values = []
        self.nu_relay_values = []
        self.psi_relay_values = []
        
        self.disturbing_moment_gamma = []
        self.disturbing_moment_psi = []
        self.disturbing_moment_nu = []
        self.phase_plane_obj = PhasePlane()

    def __system_equation(
        self, y: list, t: float
    ) -> list:
        # Система Коши | y = [gamma, psi, nu, gamma_dot, psi_dot, nu_dot]
        d_gamma_dt = y[3]
        d_psi_dt = y[4]
        d_nu_dt = y[5]
        eq = self.__get_equation_vector(y)
        d_gamma_dot_dt = eq[0, 0]
        d_psi_dot_dt = eq[1, 0]
        d_nu_dot_dt = eq[2, 0]
        return [
            d_gamma_dt,
            d_psi_dt,
            d_nu_dt,
            d_gamma_dot_dt,
            d_psi_dot_dt,
            d_nu_dot_dt,
        ]

    def __integrate_system_equation(
        self, t: float, y: list
    ) -> list:
        return self.__system_equation(y, t)

    def __get_velocity_vector(self, y) -> np.ndarray:
        return np.array([[y[3]], [y[4]], [y[5]]])

    def __get_equation_vector(self, y: list) -> np.ndarray:
        # Обращенный и простой тензор инерции
        I = ControlObject.tensor_inertia
        I_inv = np.linalg.inv(I)

        # Возмущающий и управляющий моменты
        control_moment = MotionControlSystem.control_moment

        disturbing_moment = self.__calculate_moments()

        # Вектор угловой скорости
        velocity = self.__get_velocity_vector(y)

        # Итоговое матричное уравнение
        return I_inv.dot(
            disturbing_moment
            - control_moment * self.__get_f_function_value(y)
            - np.cross(velocity.T, I.dot(velocity).T).T
        )

    def __get_initial_values(self) -> tuple:
        nu = ControlObject.get_angle_value_in_channel("nu")
        gamma = ControlObject.get_angle_value_in_channel("gamma")
        psi = ControlObject.get_angle_value_in_channel("psi")
        nu_dot = ControlObject.get_velocity_value_in_channel("nu")
        gamma_dot = ControlObject.get_velocity_value_in_channel("gamma")
        psi_dot = ControlObject.get_velocity_value_in_channel("psi")
        return gamma, psi, nu, gamma_dot, psi_dot, nu_dot

    def __get_f_function_value(self, y: list) -> np.ndarray:
        # TODO: добавить выбор нелинейной функции сигнала
        signal_nu = MotionControlSystem.linear_signal_function("nu", y[2], y[5])
        signal_gamma = MotionControlSystem.linear_signal_function("gamma", y[0], y[3])
        signal_psi = MotionControlSystem.linear_signal_function("psi", y[1], y[4])
        return np.array([
            [MotionControlSystem.f_function("gamma", signal_gamma)],
            [MotionControlSystem.f_function("psi", signal_psi)],
            [MotionControlSystem.f_function("nu", signal_nu)],
        ])

    def solve(self, end_time, step) -> None:
        # Версия решателя с постоянным шагом
        t_lst = np.linspace(0, end_time, int(end_time / step))
        sol = odeint(self.__system_equation, self.__get_initial_values(), t_lst)
        self.__set_angles_value(sol)
        self.__set_velocity_value(sol)

    def new_solve(self, end_time: float, max_step: float = 0.001,
                  rtol: float = 1e-9, atol: float = 1e-11):
        # Решатель с переменным шагом
        integrator = integrate.RK45(
            self.__integrate_system_equation,
            0.0,
            self.__get_initial_values(),
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_bound=end_time,
        )

        t_start = 0.0
        # TODO: для смены шага можно использовать:
        #if np.all(MotionControlSystem.last_value_F_function == 0):
        #    integrator.h_abs = 0.001
        while not(integrator.status == 'finished'):
            t_start = integrator.t
            integrator.step()
            curr_t = integrator.t

            # Сохранение отладочной информации
            self.__save_moment_values(
                self.__calculate_moments()
            )
            self.__save_step_size(curr_t - t_start)
            self.__save_relay_function_values()
            self.__save_step_solution(integrator.y)

            ControlObject.time_points = np.append(ControlObject.time_points, integrator.t)

    def __save_relay_function_values(self) -> None:
        self.gamma_relay_values.append(MotionControlSystem.last_value_F_function[0, 0])
        self.nu_relay_values.append(MotionControlSystem.last_value_F_function[2, 0])
        self.psi_relay_values.append(MotionControlSystem.last_value_F_function[1, 0])
    
    def __save_step_size(self, step: float) -> None:
        self.step_size_lst.append(step)
    
    def __calculate_moments(self) -> np.ndarray:
        return (
            ComputeMoments.aerodynamic_moment(self.__reduced_cosine_matrix)
            + ComputeMoments.gravitation_moment(self.__reduced_cosine_matrix)
            + ComputeMoments.magnetic_moment(self.__reduced_cosine_matrix)
            + ComputeMoments.sun_moment()
        )
    
    def __save_moment_values(self, moment: np.ndarray) -> None:
        self.disturbing_moment_gamma.append(moment[0, 0])
        self.disturbing_moment_psi.append(moment[1, 0])
        self.disturbing_moment_nu.append(moment[2, 0])
        
    def __save_step_solution(self, solution: list) -> None:
        ControlObject.set_angles_in_channel("gamma", solution.y[0])
        ControlObject.set_angles_in_channel("psi", solution.y[1])
        ControlObject.set_angles_in_channel("nu", solution.y[2])
        ControlObject.set_velocity_in_channel("gamma", solution.y[3])
        ControlObject.set_velocity_in_channel("psi", solution.y[4])
        ControlObject.set_velocity_in_channel("nu", solution.y[5])

    def __set_angles_value(self, solution: np.ndarray):
        ControlObject.set_angles_in_channel("gamma", solution[:, 0].T)
        ControlObject.set_angles_in_channel("psi", solution[:, 1].T)
        ControlObject.set_angles_in_channel("nu", solution[:, 2].T)

    def __set_velocity_value(self, solution: np.ndarray):
        ControlObject.set_velocity_in_channel("gamma", solution[:, 3].T)
        ControlObject.set_velocity_in_channel("psi", solution[:, 4].T)
        ControlObject.set_velocity_in_channel("nu", solution[:, 5].T)

    def __get_figure(self, figure_size: tuple = (5, 5)) -> plt.Axes:
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
        ax.grid(which="major", color="#DDDDDD", linewidth=1.5)
        ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=1)
        ax.minorticks_on()
        ax.grid(True)
        return ax

    def __set_borders(self, channel_name: str, ax: plt.Axes) -> dict:
        """
        Определяет границы графиков

        Parameters
        ----------
        channel_name : str
            Название канала, для которого будет отображаться фазовый портрет
        ax : plt.Axes
            Объект фигуры

        Returns
        -------
        dict
            Словарь с границами по каждой оси
        """
        x_right_border = np.max(self.data_angles_mapping[channel_name][0])
        x_left_border = np.min(self.data_angles_mapping[channel_name][0])
        y_upper_border = np.max(self.data_angles_mapping[channel_name][1])
        y_lower_border = np.min(self.data_angles_mapping[channel_name][1])

        ax.set_xlim(
            (x_left_border + 0.1 * x_left_border) * 180 / np.pi,
            (x_right_border + 0.1 * x_right_border) * 180 / np.pi,
        )
        ax.set_ylim(
            (y_lower_border + 0.1 * y_lower_border) * 180 / np.pi,
            (y_upper_border + 0.1 * y_upper_border) * 180 / np.pi,
        )

        return {
            "x_max": x_right_border,
            "x_min": x_left_border,
            "y_max": y_upper_border,
            "y_min": y_lower_border,
        }

    def __plot_coordinate_system(self, borders: dict, ax: plt.Axes) -> None:
        """
        Отображает на графике систему координат

        Parameters
        ----------
        borders : dict
            Границы графика
        ax : plt.Axes
            Объект фигуры
        """
        ax.plot(
            [0, 0],
            [
                (borders["y_min"] + 0.1 * borders["y_min"]) * 180 / np.pi,
                (borders["y_max"] + 0.1 * borders["y_max"]) * 180 / np.pi,
            ],
            color="gray",
        )
        ax.plot(
            [
                (borders["x_min"] + 0.1 * borders["x_min"]) * 180 / np.pi,
                (borders["x_max"] + 0.1 * borders["x_max"]) * 180 / np.pi,
            ],
            [0, 0],
            color="gray",
        )

    def __update_data_angles_map(self) -> None:
        """
        Обновляет ссылки на массивы с углами и скоростями
        """
        self.data_angles_mapping = {
            "nu": (
                ControlObject.nu_angles,
                ControlObject.nu_w,
            ),
            "psi": (
                ControlObject.psi_angles,
                ControlObject.psi_w,
            ),
            "gamma": (
                ControlObject.gamma_angles,
                ControlObject.gamma_w,
            ),
        }

    def plot_step_diagram(self) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("step_size", fontsize=14, fontweight="bold")
        ax.plot(
            ControlObject.time_points[:-1],
            self.step_size_lst,
            color="g",
        )

    def plot_disturbing_moment_gamma(self, angles: bool = False) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("M_gamma, Н * м", fontsize=14, fontweight="bold")
        if angles:
            x_values = ControlObject.gamma_angles[:-1]
            x_label = "gamma, рад."
        else:
            x_values= ControlObject.time_points[:-1]
            x_label = "t, c"
        plt.xlabel(x_label, fontsize=14, fontweight="bold")
        ax.plot(
            x_values,
            self.disturbing_moment_gamma,
            color="g",
        )
        plt.title("Сумм. возм. момент в крене")

    def plot_disturbing_moment_psi(self, angles: bool = False) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("M_psi, Н * м", fontsize=14, fontweight="bold")
        if angles:
            x_values = ControlObject.psi_angles[:-1]
            x_label = "psi, рад."
        else:
            x_values= ControlObject.time_points[:-1]
            x_label = "t, c"
        plt.xlabel(x_label, fontsize=14, fontweight="bold")
        ax.plot(
            x_values,
            self.disturbing_moment_psi,
            color="g",
        )
        plt.title("Сумм. возм. момент в курсе")
    
    def plot_disturbing_moment_nu(self, angles: bool = False) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("M_nu, Н * м", fontsize=14, fontweight="bold")
        if angles:
            x_values = ControlObject.nu_angles[:-1]
            x_label = "nu, рад."
        else:
            x_values= ControlObject.time_points[:-1]
            x_label = "t, c"
        plt.xlabel(x_label, fontsize=14, fontweight="bold")
        ax.plot(
            x_values,
            self.disturbing_moment_nu,
            color="g",
        )
        plt.title("Сумм. возм. момент в тангаже")

    def plot_oscillogram_gamma(self) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("gamma, рад", fontsize=14, fontweight="bold")
        ax.plot(
            ControlObject.time_points,
            ControlObject.gamma_angles,
            color="g",
        )
        plt.title("Осцилл. в крене")

    def plot_oscillogram_psi(self) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("psi, рад", fontsize=14, fontweight="bold")
        ax.plot(
            ControlObject.time_points,
            ControlObject.psi_angles,
            color="g",
        )
        plt.title("Осцилл. в курсе")

    def plot_oscillogram_nu(self) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("psi, рад", fontsize=14, fontweight="bold")
        ax.plot(
            ControlObject.time_points,
            ControlObject.nu_angles,
            color="g",
        )
        plt.title("Осцилл. в тангаже")     

    def plot_F_function_values(self) -> None:
        ax = self.__get_figure()
        plt.xlabel("t, c", fontsize=14, fontweight="bold")
        plt.ylabel("F", fontsize=14, fontweight="bold")
        ax.plot(
            ControlObject.time_points[:-1],
            self.gamma_relay_values,
            color="g",
            label="F_gamma"
        )
        ax.plot(
            ControlObject.time_points[:-1],
            self.psi_relay_values,
            color="b",
            label="F_psi"
        )
        ax.plot(
            ControlObject.time_points[:-1],
            self.nu_relay_values,
            color="r",
            label="F_nu"
        )
        ax.legend()

    def plot_phase_portrait(self, channel_name: str) -> None:
        """
        Отображает фазовый портрет

        Parameters
        ----------
        channel_name : str
            Название канала
        """
        self.__update_data_angles_map()
        ax = self.__get_figure()
        borders = self.__set_borders(channel_name, ax)
        self.__plot_coordinate_system(borders, ax)

        angles_line_1 = (
            np.linspace(
                borders["x_min"] + 0.1 * borders["x_min"],
                borders["x_max"] + 0.1 * borders["x_max"],
                10,
            )
            * 180
            / np.pi
        )
        velocity_line_1 = self.phase_plane_obj.get_values_on_switch_line(
            "L1", angles_line_1, channel_name
        )
        angles_line_2 = (
            np.linspace(
                borders["x_min"] + 0.1 * borders["x_min"],
                borders["x_max"] + 0.1 * borders["x_max"],
                10,
            )
            * 180
            / np.pi
        )
        velocity_line_2 = self.phase_plane_obj.get_values_on_switch_line(
            "L2", angles_line_2, channel_name
        )

        angles_line_3 = (
            np.linspace(
                borders["x_min"] + 0.1 * borders["x_min"],
                borders["x_max"] + 0.1 * borders["x_max"],
                10,
            )
            * 180
            / np.pi
        )
        velocity_line_3 = self.phase_plane_obj.get_values_on_switch_line(
            "L3", angles_line_3, channel_name
        )
        angles_line_4 = (
            np.linspace(
                borders["x_min"] + 0.1 * borders["x_min"],
                borders["x_max"] + 0.1 * borders["x_max"],
                10,
            )
            * 180
            / np.pi
        )
        velocity_line_4 = self.phase_plane_obj.get_values_on_switch_line(
            "L4", angles_line_4, channel_name
        )

        plt.xlabel("X, град.", fontsize=14, fontweight="bold")
        plt.ylabel("Y, град./c", fontsize=14, fontweight="bold")

        ax.plot(angles_line_1, velocity_line_1, color="g", label="L1")
        ax.plot(angles_line_2, velocity_line_2, color="b", label="L2")
        ax.plot(angles_line_3, velocity_line_3, color="g", label="L3")
        ax.plot(angles_line_4, velocity_line_4, color="b", label="L4")
        ax.plot(
            self.data_angles_mapping[channel_name][0] * 180 / np.pi,
            self.data_angles_mapping[channel_name][1] * 180 / np.pi,
            color="red",
        )
        plt.title(channel_name)

    def plot_show(self):
        plt.show()
