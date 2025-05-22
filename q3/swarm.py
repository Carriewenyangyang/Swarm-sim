from .agent import Agent
import numpy as np

# 基本框架实现。在具体做实验时看期待代码文件。原因之一是position可能会发生改变

class Swarm:
    def __init__(self, num_agents, speed=1.0, speed_list=None,
                 perception_radius=30, communication_radius=50):
        
        if speed_list is not None:
            assert len(speed_list) == num_agents, "speed_list length must match num_agents"
        self.agents = [
            Agent(
                id=i,
                position=np.random.rand(2) * 100,
                speed=speed_list[i] if speed_list is not None else speed,
                perception_radius=perception_radius
            )
            for i in range(num_agents)
        ]

        self.agents = [
            Agent(id=i,
                  position=np.random.rand(2) * 100,
                  speed=speed,
                  perception_radius=perception_radius)
            for i in range(num_agents)
        ]
        self.communication_radius = communication_radius

    def step(self, strategy="between"):
        for agent in self.agents:
            agent.broadcast()
        for agent in self.agents:
            agent.receive_broadcasts(self.agents, self.communication_radius)
            #agent.select_targets(self.agents)
            agent.select_targets_upgrade(self.agents)
            agent.update_position(strategy)

    def get_positions(self):
        return np.array([agent.position for agent in self.agents])
