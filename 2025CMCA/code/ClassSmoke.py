
import math
import numpy as np
from  ClassFly import *
G = 9.80065

class ClassSmoke():
    def __init__(self, fly:ClassFly,fly_time= 0.,fall_time=0.):
        self.fly = fly
        self.fly_time = fly_time
        self.fall_time = fall_time
        self.flag = 0
    
    # ============================================================
    # ========== 修改部分：烟雾位置计算（三个阶段） ==========
    # ============================================================
    def pos(self,t):
        # fly_time: 受领任务后到投放的时间
        # fall_time: 投放后到起爆的时间（自由落体时间）
        # 起爆时刻 = fly_time + fall_time
        # 有效遮蔽时间：起爆后20秒内
        
        if t < self.fly_time:
            # 投放前，返回无效位置
            return (0,0,-1000)
        
        # 计算无人机速度分量（等高度飞行，z方向速度为0）
        dx = self.fly.v * math.cos(self.fly.radian)
        dy = self.fly.v * math.sin(self.fly.radian)
        
        # 投放时刻的无人机位置（受领任务后fly_time秒）
        drop_x = self.fly.position[0] + dx * self.fly_time
        drop_y = self.fly.position[1] + dy * self.fly_time
        drop_z = self.fly.position[2]  # 等高度飞行
        
        if t < self.fly_time + self.fall_time:
            # 自由落体阶段（投放后到起爆前）
            dt_fall = t - self.fly_time
            # 烟幕弹在投放时具有无人机的水平速度，继续水平运动
            # 垂直方向自由落体
            smoke_x = drop_x + dx * dt_fall
            smoke_y = drop_y + dy * dt_fall
            smoke_z = drop_z - 0.5 * G * dt_fall ** 2
            return (smoke_x, smoke_y, smoke_z)
        elif t <= self.fly_time + self.fall_time + 20:
            # 起爆后云团下沉阶段（起爆后20秒内有效）
            dt_fall_total = self.fall_time
            # 起爆时刻的位置
            detonate_x = drop_x + dx * dt_fall_total
            detonate_y = drop_y + dy * dt_fall_total
            detonate_z = drop_z - 0.5 * G * dt_fall_total ** 2
            
            # 起爆后的云团位置（水平位置不变，以3m/s速度下沉）
            time_after_detonate = t - (self.fly_time + self.fall_time)
            smoke_x = detonate_x
            smoke_y = detonate_y
            smoke_z = detonate_z - 3 * time_after_detonate
            
            return (smoke_x, smoke_y, smoke_z)
        else:
            # 超过有效遮蔽时间
            return (0,0,-1000)
    # ============================================================ 
    
        
smoke_class_list: list[Type[ClassSmoke]] = []

for i in range(15):
    p = i //3
    smoke_class_list.append(ClassSmoke(fly_class_list[p]))
