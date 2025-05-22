# Agent Class
import numpy as np

class Agent:
    def __init__(self, id, position, speed, perception_radius):
        self.id = id
        self.position = np.array(position, dtype=np.float32)
        self.speed = speed
        self.perception_radius = perception_radius

        self.target_A = None
        self.target_B = None

    def set_targets(self, A, B):
        self.target_A = A
        self.target_B = B
    
    def select_targets(self, all_agents):
        # 找出所有在感知范围内的其他agent
        neighbors = [a for a in all_agents if a is not self and 
                     np.linalg.norm(self.position - a.position) <= self.perception_radius]
        if len(neighbors) >=2:
            self.target_A, self.target_B = np.random.choice(neighbors, 2, replace=False)
        else:
            self.target_A, self.target_B = None, None


    def get_direction(self, strategy="between"):
        if self.target_A is None or self.target_B is None:
            return np.zeros(2) # 如果agent没有选定A或B(也就是没有参考对象)，那就返回[0, 0], 表示不移动。

        if strategy == "between": # midpoint是目标位置，self.position是当前坐标，两者相减就是一个“方向向量”（指向中点的方向，长度可用于控制速度大小，例如（目标-当前）-》推进向目标的方向）。
            midpoint = (self.target_A.position + self.target_B.position) / 2
            return midpoint - self.position

        elif strategy == "behind": # 从A的角度藏在B的后面。
            # Try to hide behind B from A
            vector = self.target_B.position - self.target_A.position # 得到从A指向B的方向
            target = self.target_B.position + vector # 从B再往这个方向延伸一段，也就是在A->B的方向上，把B当挡箭牌，继续延伸，agent要跑到这里来躲
            return target - self.position # 算出从当前位置到这个“躲避点”的方向

        # default no movement
        return np.zeros(2)

    def update_position(self, strategy="between"):
        direction = self.get_direction(strategy) # 得知该往哪儿走，有正有负
        norm = np.linalg.norm(direction) # 求模，计算这个方向向量的模，也叫范数，或者是长度。
        if norm > 0: #当norm为零时，点停止移动。
            direction = direction / norm  # normalize， 得到了单位方向
            self.position += direction * self.speed# 决定每帧移动多少（多少个单位方向），更新agent速度


# 可选扩展： 
# 1. 如果要测试速度异质性，可以这样传：Agent(id=i, position=..., speed=np.random.uniform(0.2, 1.0), perception_radius=...)
# 2. 增加位置限制，如果你想模拟一个封闭空间内的swarm，比如边界在0-100，可以挤上这类逻辑（非必要）
# 3. 跟踪轨迹(可用于可视化agent运动轨迹)： 如果你后期展示每个agent的移动路径，可以在Agent里加一行记录历史位置。self.history = [self.position.copy()]
# self.history.append(self.position.copy())