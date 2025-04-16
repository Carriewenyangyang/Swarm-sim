# Swarm System Controller
import numpy as np
from agent import Agent

class Swarm:
    def __init__(self, num_agents=20, speed=0.5, perception_radius=50):
        self.agents = []
        self.size = 100  # 100x100 space
        self.perception_radius = perception_radius

        for i in range(num_agents):
            pos = np.random.rand(2) * self.size
            agent = Agent(i, pos, speed, perception_radius)
            self.agents.append(agent)

        # Assign random targets
        for agent in self.agents:
            A, B = np.random.choice([a for a in self.agents if a != agent], 2, replace=False)
            agent.set_targets(A, B)

    def step(self, strategy="between"):
        for agent in self.agents:
            agent.update_position(strategy)

    def get_positions(self):
        return np.array([agent.position for agent in self.agents])
