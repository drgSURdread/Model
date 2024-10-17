import numpy as np
import pandas as pd


def start_read_data(path_file: str) -> pd.DataFrame:
    """
    Считывает в pandas таблицу исходные данные для
    расчета

    Parameters
    ----------
    path_file : str
        Путь до xls файла

    Returns
    -------
    pd.DataFrame
        pandas таблица с исходными данными
    """
    return pd.read_excel(
        path_file,
        dtype={
            "tensor_inertia": float,
            "h": float,
            "k": float,
            "height": float,
            "sun_pressure_shoulder": float,
            "aero_pressure_shoulder": float,
            "magnetic_moment": float,
            "start_gamma": float,
            "start_psi": float,
            "start_nu": float,
            "start_w_gamma": float,
            "start_w_psi": float,
            "start_w_nu": float,
            "start_arg_perigee": float,
            "alpha": float,
            "control_moment_value": float,
            "channel": str,
            "inclination_orbit": float,
        },
    )


def get_tensor_inertia(file: pd.DataFrame) -> np.ndarray:
    """
    Получает из таблицы с исходными данными тензор
    инерции

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    np.ndarray
        Тензор инерции
    """
    return np.array(
        [
            [
                file.loc[0, "tensor_inertia"],
                file.loc[1, "tensor_inertia"],
                file.loc[2, "tensor_inertia"],
            ],
            [
                file.loc[3, "tensor_inertia"],
                file.loc[4, "tensor_inertia"],
                file.loc[5, "tensor_inertia"],
            ],
            [
                file.loc[6, "tensor_inertia"],
                file.loc[7, "tensor_inertia"],
                file.loc[8, "tensor_inertia"],
            ],
        ]
    )


def get_shoulders_vectors(file: pd.DataFrame) -> tuple:
    """
    Получает из таблицы с исходными данными вектора
    плеч до центров аэродинамического и солнечного давления

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    tuple = (aero_vector, sun_vector)
    """
    aerodynamic_shoulder_vector = np.array(
        [
            [file.loc[0, "aero_pressure_shoulder"]],
            [file.loc[1, "aero_pressure_shoulder"]],
            [file.loc[2, "aero_pressure_shoulder"]],
        ]
    )
    sun_pressure_shoulder_vector = np.array(
        [
            [file.loc[0, "sun_pressure_shoulder"]],
            [file.loc[1, "sun_pressure_shoulder"]],
            [file.loc[2, "sun_pressure_shoulder"]],
        ]
    )
    return aerodynamic_shoulder_vector, sun_pressure_shoulder_vector


def get_magnetic_moment(file: pd.DataFrame) -> np.ndarray:
    """
    Получает из таблицы с исходными данными собственный
    магнитный момент аппарата
    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    np.ndarray
        Вектор собственного магнитного момента аппарата
    """
    return np.array(
        [
            [file.loc[0, "magnetic_moment"]],
            [file.loc[1, "magnetic_moment"]],
            [file.loc[2, "magnetic_moment"]],
        ]
    )


def get_motion_control_param(file: pd.DataFrame) -> dict:
    """
    Получает из таблицы с исходными данными
    параметры системы управления

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    dict
        Словарь с параметрами системы управления
    """
    return {
        "h": np.array([
            [file.loc[0, "h"]],
            [file.loc[1, "h"]],
            [file.loc[2, "h"]]
        ]),
        "k": np.array([
            [file.loc[0, "k"]],
            [file.loc[1, "k"]],
            [file.loc[2, "k"]]
        ]),
        "alpha": np.array([
            [file.loc[0, "alpha"]],
            [file.loc[1, "alpha"]],
            [file.loc[2, "alpha"]]
        ]),
    }


def get_initial_values(file: pd.DataFrame) -> dict:
    """
    Получает из таблицы с исходными данными
    начальные условия углов и скоростей для
    каждого канала

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    dict
        Словарь с начальными условиями
    """
    return {
        "gamma": file.loc[0, "start_gamma"],
        "psi": file.loc[0, "start_psi"],
        "nu": file.loc[0, "start_nu"],
        "gamma_w": file.loc[0, "start_w_gamma"],
        "psi_w": file.loc[0, "start_w_psi"],
        "nu_w": file.loc[0, "start_w_nu"],
        "arg_perigee": file.loc[0, "start_arg_perigee"],
    }


def get_param_of_orbit(file: pd.DataFrame) -> tuple:
    """
    Получает из таблицы с исходными данными
    параметры орбиты

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    tuple = (height, inclination_orbit)
    """
    return file.loc[0, "height"], file.loc[0, "inclination_orbit"]


def get_control_moment(file: pd.DataFrame) -> np.ndarray:
    """
    Получает из таблицы с исходными данными
    значение управляющего момента

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    np.ndarray
        Значение управляющего момента
    """
    return np.array(
        [
            [file.loc[0, "control_moment_value"]],
            [file.loc[1, "control_moment_value"]],
            [file.loc[2, "control_moment_value"]],
        ]
    )


def get_channel_name(file: pd.DataFrame) -> str:
    """
    Получает из таблицы с исходными данными
    название исследуемого канала управления

    Parameters
    ----------
    file : pd.DataFrame
        Таблица с исходными данными

    Returns
    -------
    str
        Название канала управления

    Raises
    ------
    ValueError
        Ошибка отсутствия канала управления с
        передаваемым названием
    """
    name = file.loc[0, "channel"]
    if name not in ["nu", "psi", "gamma"]:
        raise ValueError("Канала управления с таким названием нет")
    return name
