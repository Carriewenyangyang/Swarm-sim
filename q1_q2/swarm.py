# Swarm System Controller
import numpy as np
from q1_q2.agent import Agent

class Swarm:
    def __init__(self, num_agents=20, speed=0.5, perception_radius=50):
        self.agents = []
        self.size = 100  # 100x100 space
        self.perception_radius = perception_radius

        self.history = [] # 可选：record positions in each step

        for i in range(num_agents):
            pos = np.random.rand(2) * self.size
            agent = Agent(i, pos, speed, perception_radius)
            self.agents.append(agent)

        # Assign random targets
        for agent in self.agents:
            A, B = np.random.choice([a for a in self.agents if a != agent], 2, replace=False)
            agent.set_targets(A, B)
        #agent.select_targets(self.agents)

    def step(self, strategy="between"):
        for agent in self.agents:
            agent.update_position(strategy)
        self.history.append(self.get_positions())

    def get_positions(self):
        return np.array([agent.position for agent in self.agents])
    
    def has_converged(self, thresh=5.0):
        """检测最后一次记录的swarm是否收敛到阈值内"""
        if not self.history:
            return False
        
        pos = self.history[-1]
        center = pos.mean(axis=0)
        maxd = np.max(np.linalg.norm(pos - center, axis=1))
        return maxd < thresh

    def metric(self, thresh = 5.0):
        """
        返回当前swarm的收敛状态和统计指标
        - max_radius: 所有agents到中心点的最远距离
        - avg_dispersion: 所有agents到中心点的平均距离
        - converged: 是否收敛 (最大距离小于阈值) 
        """
        positions = self.get_positions()
        center = positions.mean(axis=0)
        distances = np.linlig.norm(positions - center, axis=1)
        max_radius = np.max(distances)
        avg_dispersion = np.mean(distances)

        return {
            "max_radius": max_radius,
            "avg_dispersion": avg_dispersion,
            "converged": max_radius < thresh
        }