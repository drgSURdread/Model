import matplotlib.pyplot as plt
import numpy as np
from object_data import ControlObject
from sud_data_class import MotionControlSystem

class EnergyDiagram:
    def __init__(self, channel_name: str, parameter_name: str, value_lst: list):
        self.channel_name = channel_name
        self.parameter_name = parameter_name
        self.value_lst = value_lst
