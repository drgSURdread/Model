import numpy as np
from calculate_moments import ComputeMoments
from object_data import ControlObject


# TODO: Добавить учет отличия значений alpha, h, k в зависимости от канала
# FIXME: Вектора в alpha, h, k ломают логику работы класса PhasePlane
class MotionControlSystem:
    """
    Класс для хранения состояния системы управления

    Поля
    -----
    h : np.ndarray = np.zeros(shape=(3, 1))
        Ширина петли гистерезиса

    alpha : np.ndarray = np.zeros(shape=(3, 1))
        Уставка

    k : np.ndarray = np.zeros(shape=(3, 1))
        Коэффициент демпфирования по угловой скорости

    a : np.ndarray = np.zeros(shape=(3, 1))
        Эффективность управления

    control_moment_value : np.ndarray = np.zeros(shape=(3, 1))
        Вектор управляющего момента

    g : np.ndarray = np.zeros(shape=(3, 1))
        Эффективность возмущения

    disturbing_moment : np.ndarray = np.zeros(shape=(3, 1))
        Вектор возмущающего момента

    channel_name: str = None
        Название канала управления

    index_channel_mapping : dict
        Словарь сопоставляющий название канала и индекс в векторах
    """

    h: np.ndarray = np.zeros(shape=(3, 1))
    alpha: np.ndarray = np.zeros(shape=(3, 1))
    k: np.ndarray = np.zeros(shape=(3, 1))

    g: np.ndarray = np.zeros(shape=(3, 1))
    a: np.ndarray = np.zeros(shape=(3, 1))

    disturbing_moment: np.ndarray = np.zeros(shape=(3, 1))
    control_moment: np.ndarray = np.zeros(shape=(3, 1))

    channel_name: str = None

    index_channel_mapping: dict = {
        "gamma": 0,
        "psi": 1,
        "nu": 2,
    }

    last_value_F_function: np.ndarray = np.zeros(shape=(3, 1))

    period: float = 0.0 # Период ПЦ
    borehole: float = 0.0 # Скважность ПЦ

    @staticmethod
    def set_g_effectiveness() -> None:
        """
        Вычисляет эффективность возмущения основываясь на известном тензоре инерции
        (осевые моменты инерции) и сумме возмущающих моментов

        Parameters
        ----------
        channel_name: str
            Канал управления

        Returns
        -------
        float
           Эффективность возмущения в выбраном канале
        """
        MotionControlSystem.disturbing_moment = np.array([
            [3.63797881e-8],
            [3.63797881e-8],
            [3.63797881e-8],
        ])

        # FIXME: Моменты в канале тангажа и курса очень большие
        # надо проверить

        sum_moments = (
            ComputeMoments.aerodynamic_moment()
            + ComputeMoments.gravitation_moment()
            + ComputeMoments.magnetic_moment()
            + ComputeMoments.sun_moment()
        )

        for i in range(MotionControlSystem.g.shape[0]):
            MotionControlSystem.g[i, 0] = abs(
                sum_moments[i, 0] / ControlObject.tensor_inertia[i, i]
            )

    @staticmethod
    def set_a_effectiveness(control_moment_value: np.ndarray) -> float:
        """
        Вычисляет эффективность управления основываясь на известном тензоре инерции
        (осевые моменты инерции) в управляющем канале

        Parameters
        ----------
        control_moment_value : np.ndarray
            Управляющий момент [НМ]

        Returns
        -------
        float
           Эффективность управления
        """
        for i in range(MotionControlSystem.a.shape[0]):
            MotionControlSystem.a[i, 0] = (
                control_moment_value[i, 0] / ControlObject.tensor_inertia[i, i]
            )

        MotionControlSystem.control_moment = control_moment_value

    @staticmethod
    def get_g_in_channel(channel_name: str) -> float:
        """
        Возвращает эффективность возмущения в нужном канале
        Parameters
        ----------
        channel_name : str
            Название канала

        Returns
        -------
        float
            Значение эффективности возмущения
        """
        return MotionControlSystem.g[
            MotionControlSystem.index_channel_mapping[channel_name], 0
        ]

    @staticmethod
    def get_a_in_channel(channel_name: str) -> float:
        """
        Возвращает эффективность управления в нужном канале
        Parameters
        ----------
        channel_name : str
            Название канала

        Returns
        -------
        float
            Значение эффективности управления
        """
        return MotionControlSystem.a[
            MotionControlSystem.index_channel_mapping[channel_name], 0
        ]
    
    @staticmethod
    def check_signal_value(channel_name: str, signal_value:float) -> int:
        index_channel = MotionControlSystem.index_channel_mapping[channel_name]
        epsilon = MotionControlSystem.h[index_channel, 0] / 10
        if MotionControlSystem.alpha[index_channel, 0] - MotionControlSystem.h[index_channel, 0] - epsilon < signal_value < MotionControlSystem.alpha[index_channel, 0] + epsilon:
            return True
        elif -MotionControlSystem.alpha[index_channel, 0] + epsilon < signal_value < -MotionControlSystem.alpha[index_channel, 0] + MotionControlSystem.h[index_channel, 0] - epsilon:
            return True
        return False
    
    @staticmethod
    def f_function(channel_name: str, signal_value: float) -> int:
        """
        Метод описывающий логику трехпозиционного реле с гистерезисом и
        предысторией (закон управления)

        Parameters
        ----------
        channel_name : str
            Название канала управления
        signal_value : float
            Значение сигнала управления

        Returns
        -------
        int
            Значение сигнала управления
        """
        index_channel = MotionControlSystem.index_channel_mapping[channel_name]
        if signal_value > MotionControlSystem.alpha[index_channel, 0]:
            MotionControlSystem.last_value_F_function[index_channel, 0] = 1
        elif signal_value < -MotionControlSystem.alpha[index_channel, 0]:
            MotionControlSystem.last_value_F_function[index_channel, 0] = -1
        F = 0.5 * (
            np.sign(
                signal_value
                - MotionControlSystem.alpha[index_channel, 0]
                + MotionControlSystem.last_value_F_function[index_channel, 0]
                * MotionControlSystem.h[index_channel, 0]
            )
            + np.sign(
                signal_value
                + MotionControlSystem.alpha[index_channel, 0]
                + MotionControlSystem.h[index_channel, 0]
                * MotionControlSystem.last_value_F_function[index_channel, 0]
            )
        )
        MotionControlSystem.last_value_F_function[index_channel, 0] = F
        return F

    @staticmethod
    def linear_signal_function(
        channel_name: str, angle, velocity) -> float:
        """
        Функция вычисления значения линейного сигнала управления

        Parameters
        ----------
        channel_name : str
            Название канала управления
        angle : float
            Значение текущего угла
        velocity : float
            Значение текущей угловой скорости

        Returns
        -------
        float
            Значение сигнала управления
        """
        return (
            angle
            + MotionControlSystem.k[
                MotionControlSystem.index_channel_mapping[channel_name], 0
            ]
            * velocity
        )
