
import numpy as np
from typing import Type, Tuple  

class ClassMission:
    def __init__(self,position):
        self.position = np.array(position)
        self.flag = 0
        self._unit = self.clc_unit()
    
    def clc_unit(self):
        return -self.position / np.linalg.norm(self.position)
    def pos(self,t):
        return self.position + t * 300 * self._unit



m1 = ClassMission((20000, 0, 2000))
m2 = ClassMission((19000, 600, 2100))
m3 = ClassMission((18000, -600, 1900))

mission_class_list: list[Type[ClassMission]] = [m1,m2,m3]
