import numpy as np
from calculate_moments import ComputeMoments
from object_data import ControlObject


class MotionControlSystem:
    """
    Класс для хранения состояния системы управления

    Поля
    -----
    h : float = 0.0
        Ширина петли гистерезиса

    alpha : float = 0.0
        Уставка

    k : float = 0.0
        Коэффициент демпфирования по угловой скорости

    a : np.ndarray = np.zeros(shape=(3, 1))
        Эффективность управления

    g : np.ndarray = np.zeros(shape=(3, 1))
        Эффективность возмущения

    channel_name: str = None
        Название канала управления

    index_channel_mapping : dict
        Словарь сопоставляющий название канала и индекс в векторах
    """

    h: float = 0.0
    alpha: float = 0.0
    k: float = 0.0

    g: np.ndarray = np.zeros(shape=(3, 1))
    a: np.ndarray = np.zeros(shape=(3, 1))

    channel_name: str = None

    index_channel_mapping: dict = {
        "gamma": 0,
        "psi": 1,
        "nu": 2,
    }

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
    def set_a_effectiveness(control_moment_value: float) -> float:
        """
        Вычисляет эффективность управления основываясь на известном тензоре инерции
        (осевые моменты инерции) в управляющем канале

        Parameters
        ----------
        control_moment_value: float
            Управляющий момент [НМ]

        Returns
        -------
        float
           Эффективность управления
        """
        for i in range(MotionControlSystem.a.shape[0]):
            MotionControlSystem.a[i, 0] = (
                control_moment_value / ControlObject.tensor_inertia[i, i]
            )

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
