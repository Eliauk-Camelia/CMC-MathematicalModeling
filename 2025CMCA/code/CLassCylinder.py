
import math

class ClassCylinder:

    def __init__(self,position,r,h):

        self.position = position
        self.r = r 
        self.h = h 
        self.dx = 0.1
        self.dz = 0.5
        self.points = self.clc_points()
        self.len_points = len(self.points)
        
    def clc_points(self):
        arr = []
        x = self.position[0] - self.r 
        while x < self.position[0] + self.r :
            y0 = self.position[1] + math.sqrt(self.r ** 2 - (x - self.position[0]) ** 2)
            y1 = self.position[1] - math.sqrt(self.r ** 2 - (x - self.position[0]) ** 2)
            z = self.position[2]
            while z < self.position[2] + self.h:
                arr.append((x,y0,z))
                arr.append((x,y1,z))
                z += self.dz 
            x += self.dx 
        
        return arr


cylinder = ClassCylinder((0, 200, 0),7,10)

print(cylinder.len_points)


