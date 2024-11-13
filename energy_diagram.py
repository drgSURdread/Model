import matplotlib.pyplot as plt
import numpy as np
from object_data import ControlObject
from sud_data_class import MotionControlSystem

class EnergyDiagram:
    def __init__(self, channel_name: str, parameter_name: str, value_lst: list):
        self.channel_name = channel_name
        self.parameter_name = parameter_name
        self.value_lst = value_lst

        self.cycles = dict() # Хранит найденные на текущей итерации ПЦ
    
    def start(self, nu_matrix: np.ndarray) -> None:
        """
        Перебираем НУ в квадрате со сторонами [angle_min, angle_max] и [velocity_min, velocity_max]
        """
        for angle_start in nu_matrix[0]:
            for velocity_start in nu_matrix[1]:
                # Для каждого НУ инициализируем решатель и решаем, пока не получим цикл
                print("Начинаем движение из точки ({}, {})".format(angle_start * 180 / np.pi, velocity_start * 180 / np.pi))
                
