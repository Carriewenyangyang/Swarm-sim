import matplotlib.pyplot as plt
import matplotlib.animation as animation
from swarm import Swarm
import numpy as np

# 参数设置
num_agents = 30
strategy = "behind"
perception_radius = 30
communication_radius = 50
speed = 0.6

swarm = Swarm(num_agents, speed=speed, perception_radius=perception_radius,
              communication_radius=communication_radius)

fig, ax = plt.subplots()
scat = ax.scatter([], [], s=50, c="blue")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title(f"Swarm Strategy: {strategy} + Communication")

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(frame):
    swarm.step(strategy)
    positions = swarm.get_positions()
    scat.set_offsets(positions)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=100, init_func=init,
                              interval=100, blit=True)

plt.show()
