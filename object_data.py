import numpy as np
from numpy import cos, sin


class ControlObject:
    """
    Класс для хранения состояния объекта управления

    Поля
    -----
    height : float = 0.0
        Высота полета объекта от поверхности Земли, [км]

    inclination_orbit : float
        Наклонение орбиты, [град]

    tensor_inertia : np.ndarray[float]
        Тензор инерции объекта управления

    aerodynamic_shoulder_vector : np.ndarray[float] = [[0], [0], [0]]
        Проекции радиус-вектора от центра масс до центра аэродинамического давления, [м]

    sun_pressure_shoulder_vector : np.ndarray[float] = [[0], [0], [0]]
        Проекции радиус-вектора от центра масс до центра солнечного давления, [м]

    magnetic_moment : np.ndarray[float] = [[0], [0], [0]]
        Собственный магнитный момент аппарата, [А * м^2]

    gamma_angles : np.ndarray[float] = np.array([])
        Массив значений углов отклонения связанных осей СК
        относительно опорной в канале крена, [рад]

    psi_angles : np.ndarray[float] = np.array([])
        Массив значений углов отклонения связанных осей СК
        относительно опорной в канале курса, [рад]

    nu_angles : np.ndarray[float] = np.array([])
        Массив значений углов отклонения связанных осей СК
        относительно опорной в канале тангажа, [рад]

    gamma_w : np.ndarray[float] = np.array([])
        Массив значений угловых скоростей в канале крена, [рад/c]

    psi_w : np.ndarray[float] = np.array([])
        Массив значений угловых скоростей в канале курса, [рад/c]

    nu_w : np.ndarray[float] = np.array([])
        Массив значений угловых скоростей в канале тангажа, [рад/c]

    argument_perigee : np.ndarray[float] = np.array([])
        Массив значений аргумента перигея, [рад]

    time_points : np.ndarray[float] = np.array([])
        Массив, для хранения временных точек состояния
        объекта, [c]
    """

    height: float = 0.0
    inclination_orbit: float = 0.0

    tensor_inertia: np.ndarray[float] = np.zeros(shape=(3, 3))

    aerodynamic_shoulder_vector: np.ndarray[float] = np.zeros(shape=(3, 1))
    sun_pressure_shoulder_vector: np.ndarray[float] = np.zeros(shape=(3, 1))
    magnetic_moment: np.ndarray[float] = np.zeros(shape=(3, 1))

    gamma_angles: list = list()
    psi_angles: list = list()
    nu_angles: list = list()
    argument_perigee: list = list()

    gamma_w: list = list()
    psi_w: list = list()
    nu_w: list = list()

    time_points: list = [0.0]

    # Далее идут данные для вычисления различных характеристик циклов
    y_L1: list = [] # Значения скорости на L1

    @staticmethod
    def set_angles_in_channel(channel_name: str, value: float) -> None:
        """
        Сохраняет значение угла в нужном канале
        Parameters
        ----------
        channel_name : str
            Название канала
        value : float
            Значение угла
        """
        if channel_name == "gamma":
            # ControlObject.gamma_angles = np.append(ControlObject.gamma_angles, value)
            ControlObject.gamma_angles.append(value)
        elif channel_name == "psi":
            # ControlObject.psi_angles = np.append(ControlObject.psi_angles, value)
            ControlObject.psi_angles.append(value)
        else:
            # ControlObject.nu_angles = np.append(ControlObject.nu_angles, value)
            ControlObject.nu_angles.append(value)
        
    @staticmethod
    def set_velocity_in_channel(channel_name: str, value: float) -> None:
        """
        Сохраняет значение угловой скорости в нужном канале
        Parameters
        ----------
        channel_name : str
            Название канала
        value : float
            Значение угловой скорости
        """
        if channel_name == "gamma":
            # ControlObject.gamma_w = np.append(ControlObject.gamma_w, value)
            ControlObject.gamma_w.append(value)
        elif channel_name == "psi":
            # ControlObject.psi_w = np.append(ControlObject.psi_w, value)
            ControlObject.psi_w.append(value)
        else:
            # ControlObject.nu_w = np.append(ControlObject.nu_w, value)
            ControlObject.nu_w.append(value)

    @staticmethod
    def get_angle_value_in_channel(channel_name: str) -> float:
        """
        Возвращает последнее рассчитанное значение угла
        Parameters
        ----------
        channel_name : str
            Название канала

        Returns
        -------
        float
            Значение угла
        """
        if channel_name == "gamma":
            value = ControlObject.gamma_angles[-1]
        elif channel_name == "psi":
            value = ControlObject.psi_angles[-1]
        else:
            value = ControlObject.nu_angles[-1]
        return value

    @staticmethod
    def get_velocity_value_in_channel(channel_name: str) -> float:
        """
        Возвращает последнее рассчитанное значение угловой скорости
        Parameters
        ----------
        channel_name : str
            Название канала

        Returns
        -------
        float
            Значение угловой скорости
        """
        if channel_name == "gamma":
            value = ControlObject.gamma_w[-1]
        elif channel_name == "psi":
            value = ControlObject.psi_w[-1]
        else:
            value = ControlObject.nu_w[-1]
        return value

    @staticmethod
    def get_matrix_of_guiding_cosines(
        reduced: bool = False,
    ) -> np.ndarray[float, (3, 3)]:
        """
        Вычисляет либо полную либо редуцированную матрицу направляющих косинусов

        Parameters
        ----------
        reduced : bool = False
            Отвечает за возвращение редуцированной матрицы направляющих косинусов

        Returns
        -------
        np.ndarray[float, (3, 3)]
            Матрица направляющих косинусов для текущего отклонения осей
            связанной СК относительно опорной
        """
        gamma = ControlObject.gamma_angles[-1] * np.pi / 180
        psi = ControlObject.psi_angles[-1] * np.pi / 180
        nu = ControlObject.nu_angles[-1] * np.pi / 180

        if reduced:
            return np.array([[1, nu, -psi], [-nu, 1, gamma], [psi, -gamma, 1]])

        return np.array(
            [
                [cos(nu) * cos(psi), sin(nu) * cos(psi), -sin(psi)],
                [
                    -cos(gamma) * sin(nu) + sin(gamma) * sin(psi) * cos(nu),
                    cos(gamma) * cos(nu) + sin(gamma) * sin(psi) * sin(nu),
                    sin(gamma) * cos(psi),
                ],
                [
                    cos(gamma) * sin(psi) * cos(nu) + sin(gamma) * sin(nu),
                    cos(gamma) * sin(psi) * sin(nu) - sin(gamma) * cos(nu),
                    cos(gamma) * cos(psi),
                ],
            ]
        )
