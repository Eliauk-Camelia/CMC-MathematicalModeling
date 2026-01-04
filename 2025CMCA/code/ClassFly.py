
import math
from typing import Type, Tuple  
class ClassFly:

    def __init__(self,position,v = 0.,angle = 0.):
        self.position = position
        self.v = v
        self.angle = angle

    @property
    def radian(self):
        return self.angle * math.pi / 180.0
    

fly1 = ClassFly((17800, 0, 1800))
fly2 = ClassFly((12000, 1400, 1400))
fly3 = ClassFly((6000, -3000, 700))
fly4 = ClassFly((11000, 2000, 1800))
fly5 = ClassFly((13000, -2000, 1300))

fly_class_list: list[Type[ClassFly]] = [fly1, fly2, fly3, fly4, fly5]
    

