import numpy as np
from sud_data_class import MotionControlSystem


class PhasePlane:
    """
    Класс для работы с фазовой плоскостью

    Поля
    ---------
    channel_name : str
        Название канала управления для которого осуществляется
        решение
    current_curve : str
        Текущая фазовая траектория
    lines_mapping : dict
        Словарь со ссылками. Именам линий переключения соответствуют
        функции для работы с ними
    """

    def __init__(self):
        self.lines_mapping = {
            "L1": self.__switching_line_1,
            "L2": self.__switching_line_2,
            "L3": self.__switching_line_3,
            "L4": self.__switching_line_4,
        }

    def __switching_line_1(self, channel_name, angle_value: float) -> float:
        """
        Линия переключения L1
        Parameters
        ----------
        angle_value : float
            Значение угла

        Returns
        -------
        float
            Значение функции линии переключения
        """
        index = MotionControlSystem.index_channel_mapping[channel_name]
        return (MotionControlSystem.alpha[index, 0] - angle_value) / MotionControlSystem.k[index, 0]

    def __switching_line_2(self, channel_name, angle_value: float) -> float:
        """
        Линия переключения L2
        Parameters
        ----------
        angle_value : float
            Значение угла

        Returns
        -------
        float
            Значение функции линии переключения
        """
        index = MotionControlSystem.index_channel_mapping[channel_name]
        return (
            MotionControlSystem.alpha[index, 0] - MotionControlSystem.h[index, 0] - angle_value
        ) / MotionControlSystem.k[index, 0]

    def __switching_line_3(self, channel_name, angle_value: float) -> float:
        """
        Линия переключения L3
        Parameters
        ----------
        angle_value : float
            Значение угла

        Returns
        -------
        float
            Значение функции линии переключения
        """
        index = MotionControlSystem.index_channel_mapping[channel_name]
        return (-MotionControlSystem.alpha[index, 0] - angle_value) / MotionControlSystem.k[index, 0]

    def __switching_line_4(self, channel_name, angle_value: float) -> float:
        """
        Линия переключения L4
        Parameters
        ----------
        angle_value : float
            Значение угла

        Returns
        -------
        float
            Значение функции линии переключения
        """
        index = MotionControlSystem.index_channel_mapping[channel_name]
        return (
            -MotionControlSystem.alpha[index, 0] + MotionControlSystem.h[index, 0] - angle_value
        ) / MotionControlSystem.k[index, 0]

    def get_values_on_switch_line(self, line_name: str, values, channel):
        """
        Возвращает значение функции линии переключения
        Parameters
        ----------
        line_name : str
            Название линии переключения
        values
            Значение угла

        Returns
        -------
        Значение функции линии переключения в градусах
        """
        return self.lines_mapping[line_name](channel, values * np.pi / 180) * 180 / np.pi

    def init_start_list(self, point: tuple, channel_name: str) -> None:
        """
        Инициализирует лист на фазовой плоскости, соответствующий
        переданному начальному условию
        Parameters
        ----------
        point : tuple
            Точка на фазовой плоскости
        channel_name : str
            Название канала
        """
        self.channel_name = channel_name

        if point[1] > self.__switching_line_1(channel_name, point[0]):
            self.current_curve = "G+"
        elif point[1] < self.__switching_line_3(channel_name, point[0]):
            self.current_curve = "G-"
        else:
            self.current_curve = "G0"

    def check_intersection_line_with_new_step(self, curr_point: tuple) -> bool:
        """
        Проверяет пересечение фазовой траектории с линией переключения
        Parameters
        ----------
        curr_point : tuple
            Текущая точка на фазовой плоскости

        Returns
        -------
        bool
            Флаг пересечения
        """
        if self.current_curve == "G+":
            projection_point = self.lines_mapping["L2"](self.channel_name, curr_point[0])
            if projection_point < curr_point[1]:
                return False, "L2"
            return True, "L2"
        elif self.current_curve == "G-":
            projection_point = self.lines_mapping["L4"](self.channel_name, curr_point[0])
            if projection_point > curr_point[1]:
                return False, "L4"
            return True, "L4"
        else:
            if curr_point[0] > 0:
                projection_point = self.lines_mapping["L1"](self.channel_name, curr_point[0])
                if projection_point > curr_point[1]:
                    return False, "L1"
                return True, "L1"
            else:
                projection_point = self.lines_mapping["L3"](self.channel_name, curr_point[0])
                if projection_point < curr_point[1]:
                    return False, "L3"
                return True, "L3"

    def update_current_curve(self, line_intersection: str) -> None:
        """
        Обновляет имя фазовой траектории после пересечения
        Parameters
        ----------
        line_intersection : str
            Название линии переключения с которой произошло
            пересечение
        """
        if line_intersection == "L1":
            self.current_curve = "G+"
        elif line_intersection == "L2" or line_intersection == "L4":
            self.current_curve = "G0"
        else:
            self.current_curve = "G-"

    def get_next_point(self, curr_point: tuple, step: float) -> tuple:
        """
        Возвращает следующую точку на фазовой плоскости
        Parameters
        ----------
        curr_point : tuple
            Текущая точка на фазовой плоскости
        step : float
            Шаг по времени

        Returns
        -------
        tuple
            Координаты новой точки на фазовой плоскости
        """
        if self.current_curve == "G+":
            next_point_x = (
                curr_point[0]
                + curr_point[1] * step
                - (
                    MotionControlSystem.get_a_in_channel(self.channel_name)
                    - MotionControlSystem.get_g_in_channel(self.channel_name)
                )
                * step**2
                / 2
            )
            next_point_y = (
                curr_point[1]
                - (
                    MotionControlSystem.get_a_in_channel(self.channel_name)
                    - MotionControlSystem.get_g_in_channel(self.channel_name)
                )
                * step
            )
        elif self.current_curve == "G-":
            next_point_x = (
                curr_point[0]
                + curr_point[1] * step
                + (
                    MotionControlSystem.get_a_in_channel(self.channel_name)
                    + MotionControlSystem.get_g_in_channel(self.channel_name)
                )
                * step**2
                / 2
            )
            next_point_y = (
                curr_point[1]
                + (
                    MotionControlSystem.get_a_in_channel(self.channel_name)
                    + MotionControlSystem.get_g_in_channel(self.channel_name)
                )
                * step
            )
        else:  # self.current_curve == "G0"
            next_point_x = (
                curr_point[0]
                + curr_point[1] * step
                + MotionControlSystem.get_g_in_channel(self.channel_name) * step**2 / 2
            )
            next_point_y = (
                curr_point[1]
                + MotionControlSystem.get_g_in_channel(self.channel_name) * step
            )
        return next_point_x, next_point_y
