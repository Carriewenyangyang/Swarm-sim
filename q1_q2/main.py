# Main entry: Run simulation.
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from swarm import Swarm
import numpy as np

# Create swarm
swarm = Swarm(num_agents=30, speed=0.5, perception_radius=50)
strategy = "behind" #"between"  # or "behind"
converged_frame = None

fig, ax = plt.subplots()
scat = ax.scatter([], [], s=50)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title(f"Swarm Strategy: {strategy}")

def init():
    scat.set_offsets(np.empty((0,2)))
    return scat,

def update(frame):
    swarm.step(strategy) # 计算更新agent的位置
    positions = swarm.get_positions() # 获取agent的位置
    scat.set_offsets(positions) #感觉是更新位置，查查TODO
    return scat,

ani = animation.FuncAnimation(fig, update, frames=100, init_func=init,
                              interval=100, blit=True)

plt.show()
