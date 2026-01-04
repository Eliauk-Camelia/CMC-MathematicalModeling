from CLassCylinder import *
from ClassFly import *
from ClassMission import *
from ClassSmoke import *

file_list = [[],[],[]]
total_time = []
# ============================================================
# ========== 修改部分：全局变量初始化 ==========
# ============================================================
def init():
    global file_list, total_time
    for i in range(3):
        mission_class_list[i].flag = 0
    
    for i in range(15):
        smoke_class_list[i].flag = 0
    
    # 清空全局变量
    for i in range(3):
        file_list[i].clear()
    total_time.clear()
# ============================================================


def clc_start_time():
    start_time = 66
    for i in range(15):
        if smoke_class_list[i].flag == 0:
            continue
        if start_time > smoke_class_list[i].fly_time + smoke_class_list[i].fall_time:
            start_time = smoke_class_list[i].fly_time + smoke_class_list[i].fall_time
    return start_time

def clc_end_time():
    end_time = 0
    for i in range(15):
        if smoke_class_list[i].flag == 0:
            continue
        if end_time < smoke_class_list[i].fly_time + smoke_class_list[i].fall_time + 20:
            end_time = smoke_class_list[i].fly_time + smoke_class_list[i].fall_time + 20
    return end_time

def clc_distance(a,b):
    return np.linalg.norm(np.array(a) - np.array(b))

import numpy as np

def clc_line_distance(a, b, c):
    """
    【三维专属】计算点C到直线AB的垂直距离
    :param a: 直线AB上的点A，三维坐标（列表/元组，如(1,2,3)）
    :param b: 直线AB上的点B，三维坐标（同a）
    :param c: 待计算的点C，三维坐标（同a）
    :return: 点C到直线AB的距离（浮点数）
    """
    # 1. 转换为numpy三维数组（确保输入是3维）
    vec_a = np.array(a, dtype=np.float64)
    vec_b = np.array(b, dtype=np.float64)
    vec_c = np.array(c, dtype=np.float64)
    
    # 校验输入维度（避免传入非三维点）
    if vec_a.shape[0] != 3 or vec_b.shape[0] != 3 or vec_c.shape[0] != 3:
        raise ValueError("输入必须是三维坐标！如(0,0,0)、(1,2,3)")
    
    # 2. 计算核心向量：AB = B - A，AC = C - A
    vec_ab = vec_b - vec_a  # 对应你代码中的 n1 = a - b（仅方向相反，模长一致）
    vec_ac = vec_c - vec_a  # 对应你代码中的 n2 = a - c（仅方向相反，模长一致）
    
    # 3. 边界处理：若A、B重合（直线不存在），返回C到A的欧式距离
    ab_norm = np.linalg.norm(vec_ab)  # 计算AB向量的模长
    if ab_norm < 1e-8:  # 浮点精度容错，避免除以0
        return np.linalg.norm(vec_ac)
    
    # 4. 三维向量叉乘 + 距离计算（核心步骤）
    cross_product = np.cross(vec_ab, vec_ac)  # 三维叉乘直接计算，无需扩展
    cross_norm = np.linalg.norm(cross_product)  # 叉乘结果的模长
    distance = cross_norm / ab_norm  # 最终距离
    
    return distance

# ============================================================
# ========== 修改部分：遮挡判断逻辑（精确投影参数检查） ==========
# ============================================================
def clc():
    start_time = clc_start_time()
    end_time = clc_end_time()
    t = start_time
    dt = 0.01
    effective_radius = 10.0
    R_sq = effective_radius * effective_radius
    
    while t <= end_time:
        for i in range(3):
            if mission_class_list[i].flag == 0:
                continue
            mission_pos = mission_class_list[i].pos(t)
            flag = 1  # 假设完全遮挡
            
            # 检查圆柱体上的每个点是否被遮挡
            for j in range(cylinder.len_points):
                target_point = np.array(cylinder.points[j])
                flags = 0  # 该点是否被遮挡
                
                # 检查所有烟雾是否遮挡该点
                for k in range(15):
                    if smoke_class_list[k].flag == 0:
                        continue
                    smoke_pos = smoke_class_list[k].pos(t)
                    if smoke_pos[2] == -1000:
                        continue
                    
                    smoke_center = np.array(smoke_pos)
                    
                    # 计算从导弹到目标点的向量
                    v = target_point - mission_pos
                    v_len_sq = np.dot(v, v)
                    
                    if v_len_sq < 1e-10:  # 避免数值误差，导弹和目标重合
                        # 导弹和目标重合的特殊情况
                        distance = np.linalg.norm(smoke_center - mission_pos)
                        if distance <= effective_radius:
                            flags = 1
                            break
                    else:
                        # 计算烟幕中心到直线的距离
                        w = smoke_center - mission_pos
                        t_proj = np.dot(w, v) / v_len_sq
                        
                        # 计算烟幕中心到直线的最近点
                        closest_point = mission_pos + t_proj * v
                        
                        # 烟幕中心到直线的距离
                        distance = np.linalg.norm(smoke_center - closest_point)
                        
                        # 检查距离是否在有效半径内
                        if distance <= effective_radius:
                            # 检查烟幕是否在连线上（投影参数在[0,1]之间）
                            # 或者导弹在烟幕内
                            inside_smoke = np.linalg.norm(mission_pos - smoke_center) <= effective_radius
                            
                            if inside_smoke or (0.0 <= t_proj <= 1.0):
                                flags = 1
                                break
                
                # 如果该点未被遮挡，则不完全遮挡
                if flags == 0:
                    flag = 0
                    break
            
            # 如果完全遮挡，记录该时刻
            if flag == 1:
                file_list[i].append(t)
        t += dt
    
    for i in range(3):
        x = 0.01 * len(file_list[i])
        total_time.append(x)
# ============================================================


            