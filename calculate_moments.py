import constants as const
import numpy as np
from object_data import ControlObject


class ComputeMoments:
    @staticmethod
    def gravitation_moment(reduced: bool = False) -> np.ndarray[float]:
        """
        Вычисляет момент действующий на КА от сил гравитационного взаимодействия

        Returns
        -------
        np.ndarray[float, (1, 3)]
            Вектор момента гравитационного взаимодействия [НМ]
        """
        u_velocity = np.sqrt(
            const.MU / ((const.RADIUS_OF_EARTH + ControlObject.height) * 1000)**3
        )  # Угловая скорость рад/сек
        # Перенесенный вектор местной вертикали ОСК в ССК np.array[float, (1, 3)]
        ort_ssk = np.dot(ControlObject.get_matrix_of_guiding_cosines(reduced=reduced), [[0], [1], [0]])
        return np.cross(
            (3 * u_velocity**2 * ort_ssk).T,
            np.dot(ControlObject.tensor_inertia, (ort_ssk)).T
        ).T

    @staticmethod
    def magnetic_moment(reduced: bool = False) -> np.ndarray[float]:
        """
        Вычисляет момент действующий на КА от сил магнитного взаимодействия

        Returns
        -------
        np.ndarray[float, (1, 3)]
            Вектор момента магнитного взаимодействия [НМ]
        """
        b_mag_ind = np.array(
            [
                [
                    (
                        np.cos(ControlObject.argument_perigee[-1])
                        * np.sin((ControlObject.inclination_orbit) * np.pi / 180)
                    )
                    * const.INDUCTION_OF_MAGNETIC
                    * ((ControlObject.height / const.RADIUS_OF_EARTH) ** -3)
                ],
                [
                    (
                        -2
                        * np.sin(ControlObject.inclination_orbit * np.pi / 180)
                        * np.sin(ControlObject.argument_perigee[-1])
                    )
                    * const.INDUCTION_OF_MAGNETIC
                    * ((ControlObject.height / const.RADIUS_OF_EARTH) ** -3)
                ],
                [
                    -np.cos(ControlObject.inclination_orbit * np.pi / 180)
                    * const.INDUCTION_OF_MAGNETIC
                    * ((ControlObject.height / const.RADIUS_OF_EARTH) ** -3)
                ],
            ]
        )
        return np.cross(
            (ControlObject.get_matrix_of_guiding_cosines(reduced).dot(ControlObject.magnetic_moment)).T,
            (const.INDUCTION_OF_MAGNETIC * (b_mag_ind)).T
        ).T

    @staticmethod
    def aerodynamic_moment(reduced: bool = False) -> np.ndarray[float]:
        """
        Вычисляет момент действующий на КА от сил аэродинамического взаимодействия

        Returns
        -------
        np.ndarray[float, (1, 3)]
            Вектор момента аэродинамического взаимодействия [Н*м]
        """ ""
        # Квадрат линейной скорости [м^2/с^2]
        velocity_square = const.MU / (const.RADIUS_OF_EARTH + ControlObject.height)
        # Скоростной напор [кг/м^3 * м^2/с^2]
        q = 0.5 * const.RO_ATM * velocity_square
        # Вектор аэродинамической силы [Н]
        q_force = ControlObject.get_matrix_of_guiding_cosines(reduced).dot(
            np.array([[q * const.AREA_MIDDLE * const.BALLISTIC_COEF], [0], [0]])
        )
        return np.cross(-ControlObject.aerodynamic_shoulder_vector, q_force, axis=0)

    @staticmethod
    def sun_moment() -> np.ndarray[float]:
        """
        Вычисляет момент действующий на КА от силы солнечного давления

        Returns
        -------
        np.ndarray[float, (1, 3)]
            Вектор момента солнечного давления [НМ]
        """
        # Вектор сил солнечного давления [Н]
        f_force = np.array(
            [
                [
                    const.E0
                    / const.SUN_VELOCITY
                    * const.SUN_AREA
                    * (1 + const.REFLECTION_COEF)
                ],
                [0],
                [0],
            ]
        )
        return np.cross(-ControlObject.sun_pressure_shoulder_vector, f_force, axis=0)
