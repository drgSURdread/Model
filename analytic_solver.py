import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from object_data import ControlObject
from phase_plane import PhasePlane


# TODO: добавить расчет аргумента перигея
class AnalyticSolver:
    """
    Класс аналитического решателя. Решение осуществляется
    методом фазовой плоскости по аналитическим зависимостям.

    Поля
    ---------
    phase_plane_obj : PhasePlane
        Объект фазовой плоскости на которой происходит решение

    control_channel : str
        Название канала управления для которого осуществляется
        решение

    data_angles_mapping : dict
        Словарь со ссылками. Именам каналов соответствуют ссылки
        на массивы с углами и скоростями, полученными в ходе решения
    """

    def __init__(
        self,
        control_channel: str = "nu",
    ):
        self.phase_plane_obj = PhasePlane()

        self.__check_valid_control_channel(control_channel)
        self.__init_position()

    def __check_valid_control_channel(self, control_channel: str) -> None:
        """
        Проверяет правильность названия переданного канала управления

        Parameters
        ----------
        control_channel : str
            Название канала управления
        """
        if control_channel.lower() in {"nu", "psi", "gamma"}:
            self.control_channel = control_channel
        else:
            raise ValueError("control_channel not in {nu, psi, gamma}")

    def __init_position(self) -> None:
        """
        Инициализирует лист на фазовой плоскости для начальных условий

        Parameters
        ----------
        nu : tuple
            Начальные условия (угол, скорость)
        """

        angle_point = ControlObject.get_angle_value_in_channel(self.control_channel)
        velocity_point = ControlObject.get_velocity_value_in_channel(
            self.control_channel
        )
        # Определяем текущую кривую для начального условия
        self.phase_plane_obj.init_start_list(
            channel_name=self.control_channel, point=(angle_point, velocity_point)
        )

    def __set_angle_value(self, value: float) -> None:
        """
        Сохраняет рассчитанное значение угла в объекте управления

        Parameters
        ----------
        value : float
            Рассчитанное значение угла
        """
        ControlObject.set_angles_in_channel(self.control_channel, value)

    def __set_velocity_value(self, value: float) -> None:
        """
        Сохраняет рассчитанное значение скорости в объекте управления

        Parameters
        ----------
        value : float
            Рассчитанное значение угловой скорости
        """
        ControlObject.set_velocity_in_channel(self.control_channel, value)

    def __set_time_value(self, step: float) -> None:
        """
        Сохраняет значение времени в объекте управления
        при котором производился расчет

        Parameters
        ----------
        step : float
            Шаг по времени
        """
        ControlObject.time_points = np.append(
            ControlObject.time_points, ControlObject.time_points[-1] + step
        )

    def solve(
        self, dt_max: float = 0.01, tolerance: float = 2e-8, count_steps: int = 1, step_solver: bool = False,
        time_solve: float = 10.0
    ) -> None:
        """
        Функция решения. Решение производится в count_steps шагов.
        При повторном запуске данной функции, решение запускается с
        последней выполненной итерации

        Parameters
        ----------
        dt_max : float, optional
            Максимальный шаг по времени, by default 0.01
        tolerance : float, optional
            Точность, by default 2e-8
        count_steps : int, optional
            Количество шагов решения, by default 1
        step_solver : bool, optional
            Запустить решатель по шагам, by default False
        time_solve: float, optional
            Время непрерывного решения, by default 10.0

        Raises
        ------
        ValueError
            Большое количество шагов (зависание решателя)
        """
        if step_solver:
            self.__step_solver(
                dt_max=dt_max,
                tolerance=tolerance,
                count_steps=count_steps,
            )
        else:
            self.__continuous_solver(
                dt_max=dt_max,
                tolerance=tolerance,
                time_solve=time_solve,
            )

    def __step_solver(self, dt_max: float = 0.01, tolerance: float = 2e-8, count_steps: int = 1) -> None:
        """
        Функция решения. Решение производится в count_steps шагов.

        Parameters
        ----------
        dt_max : float, optional
            Максимальный шаг по времени, by default 0.01
        tolerance : float, optional
            Точность, by default 2e-8
        count_steps : int, optional
            Количество шагов решения, by default 1

        Raises
        ------
        ValueError
            Большое количество шагов (зависание решателя)
        """
        step_time = dt_max
        for _ in range(count_steps):
            current_angle = ControlObject.get_angle_value_in_channel(
                self.control_channel
            )
            current_w = ControlObject.get_velocity_value_in_channel(
                self.control_channel
            )
            steps_count = 0
            while True:
                if steps_count > 1e5:
                    # FIXME: При моделировании разворотов, решатель всегда будет зависать
                    raise ValueError("Решатель завис, понизьте точность")

                current_angle, current_w, step_time, intersection = self.__next_step(
                    curr_point=((current_angle, current_w)),
                    step=step_time,
                    tolerance=tolerance,
                )
                if intersection:
                    break
                self.__set_angle_value(current_angle)
                self.__set_velocity_value(current_w)
                self.__set_time_value(step_time)

                steps_count += 1

            # После прохождения линии переключения
            # возвращаемся к дефолтному шагу
            step_time = dt_max
            steps_count = 0
    
    def __continuous_solver(self, dt_max: float = 0.01, tolerance: float = 2e-8, time_solve: float = 10.0) -> None:
        """
        Функция решения. Решение проходит непрерывно, пока не пройдет требуемое время `time_solve`.

        Parameters
        ----------
        dt_max : float, optional
            Максимальный шаг по времени, by default 0.01
        tolerance : float, optional
            Точность, by default 2e-8
        time_solve : float, optional
            Время решения, by default 10.0
        """
        count_step_progress = 0
        step_time = dt_max
        while ControlObject.time_points[-1] < time_solve:
            if isclose(ControlObject.time_points[-1], count_step_progress * time_solve / 10, rel_tol=2e-3):
                count_step_progress += 1
                print("Прогресс выполнения: [", "#" * count_step_progress, "]")

            current_angle = ControlObject.get_angle_value_in_channel(
                self.control_channel
            )
            current_w = ControlObject.get_velocity_value_in_channel(
                self.control_channel
            )

            current_angle, current_w, step_time, intersection = self.__next_step(
                curr_point=((current_angle, current_w)),
                step=step_time,
                tolerance=tolerance,
                dt_max=dt_max,
            )

            self.__set_angle_value(current_angle)
            self.__set_velocity_value(current_w)
            self.__set_time_value(step_time)

            if intersection:
                step_time = dt_max

    def __next_step(self, curr_point: tuple, step: float, tolerance: float, dt_max: float = 0.05) -> tuple:
        """
        Выполняет следующий шаг аналитического решения

        Parameters
        ----------
        curr_point : tuple
            Текущая точка (угол, скорость) на фазовой плоскости
        step : float
            Шаг по времени
        tolerance : float
            Точность

        Returns
        -------
        tuple
            (next_angle, next_w, step_time, intersection_flag)
        """
        step_time, intersection, current_line = self.__set_new_step_time(curr_point, step, tolerance)

        if intersection:
            if self.phase_plane_obj.current_curve == "G+":
                if len(ControlObject.y_L1) == 0:
                    # ControlObject.y_L1 = np.append(ControlObject.y_L1, curr_point[1])
                    ControlObject.y_L1.append(curr_point[1])
                else:
                    distance = abs(curr_point[1] - ControlObject.y_L1[-1])
                    # ControlObject.y_L1 = np.append(ControlObject.y_L1, curr_point[1])
                    ControlObject.y_L1.append(curr_point[1])
                    if distance < 1e-7:
                        print('Попали в предельный цикл')
            return curr_point[0], curr_point[1], step_time, True

        next_angle, next_w = self.phase_plane_obj.get_next_point(
            curr_point=curr_point, step=step_time
        )
        return next_angle, next_w, step_time, False

    def __set_new_step_time(
        self, curr_point: tuple, curr_step: float, tolerance: float, dt_max: float = 0.05,
    ) -> tuple:
        """
        Устанавливает новый шаг по времени

        Parameters
        ----------
        curr_point : tuple
            Текущая точка (угол, скорость) на фазовой плоскости
        curr_step : float
            Текущий шаг по времени
        tolerance : float
            Точность

        Returns
        -------
        tuple
            (new_step_time, intersection_flag)
        """
        new_step_time = curr_step
        while True:
            next_step_angle, next_step_w = self.phase_plane_obj.get_next_point(
                curr_point=curr_point, step=new_step_time
            )
            intersection, line = (
                self.phase_plane_obj.check_intersection_line_with_new_step(
                    curr_point=(next_step_angle, next_step_w)
                )
            )

            if not intersection:
                # Если на текущем шаге и удвоенном, не произошло пересечение, то увеличиваем шаг
                next_step_angle, next_step_w = self.phase_plane_obj.get_next_point(
                    curr_point=curr_point, step=new_step_time * 2
                )
                intersection, line = (
                    self.phase_plane_obj.check_intersection_line_with_new_step(
                        curr_point=(next_step_angle, next_step_w)
                    )
                )
                if not intersection and 2 * new_step_time <= dt_max:
                    return new_step_time * 2, False, line
                else:
                    return new_step_time, False, line

            if new_step_time < tolerance:
                self.phase_plane_obj.update_current_curve(line)
                return new_step_time, True, line

            new_step_time = new_step_time / 2

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
                np.array(ControlObject.nu_angles),
                np.array(ControlObject.nu_w),
            ),
            "psi": (
                np.array(ControlObject.psi_angles),
                np.array(ControlObject.psi_w),
            ),
            "gamma": (
                np.array(ControlObject.gamma_angles),
                np.array(ControlObject.gamma_w),
            ),
        }

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
            "L1", angles_line_1, self.control_channel,
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
            "L2", angles_line_2, self.control_channel,
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
            "L3", angles_line_3, self.control_channel,
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
            "L4", angles_line_4, self.control_channel,
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
        # plt.savefig("dz/phase_portrait/src")
        plt.show()

    def plot_x_oscillogram(self, channel_name: str) -> None:
        """
        Отображает осциллограмму угловой координаты

        Parameters
        ----------
        channel_name : str
            Название канала
        """
        self.__update_data_angles_map()
        ax1 = self.__get_figure()
        borders = self.__set_borders(channel_name, ax1)
        self.__plot_coordinate_system(borders, ax1)

        plt.xlabel("t, с", fontsize=14, fontweight="bold")
        plt.ylabel("X, град.", fontsize=14, fontweight="bold")

        ax1.plot(
            ControlObject.time_points,
            self.data_angles_mapping[channel_name][0] * 180 / np.pi,
            color="g",
        )

        plt.show()

    def plot_y_oscillogram(self, channel_name: str) -> None:
        """
        Отображает осциллограмму угловой скорости

        Parameters
        ----------
        channel_name : str
            Название канала
        """
        self.__update_data_angles_map()
        ax2 = self.__get_figure()
        borders = self.__set_borders(channel_name, ax2)
        self.__plot_coordinate_system(borders, ax2)
        plt.xlabel("t, с", fontsize=14, fontweight="bold")
        plt.ylabel("Y, град./c", fontsize=14, fontweight="bold")

        ax2.plot(
            ControlObject.time_points,
            self.data_angles_mapping[channel_name][1] * 180 / np.pi,
            color="g",
        )

        plt.show()
