import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置是否启用通信
use_comm = True

if use_comm:
    from q3.swarm import Swarm # 有通信版本(Q3)

else:
    from q1_q2.swarm import Swarm #无通信版本 （Q1/Q2）

# 统一的收敛检测函数
def has_converged(positions, thresh=5.0):
    center = positions.mean(axis=0)
    maxd = np.max(np.linalg.norm(positions - center, axis=1))
    return maxd < thresh

# 跑一轮实验，返回收敛所需步数或None
def run_once(perception_radius=30, communication_radius=None,num_agents=30, 
             speed=0.5, max_steps=300, strategy="between", seed=None):
    if seed is not None:
        np.random.seed(seed)
    if use_comm:
        swarm = Swarm(num_agents=num_agents,
                      speed=speed,
                      perception_radius=perception_radius,
                      communication_radius=communication_radius)
    else:
        swarm = Swarm(num_agents=num_agents,
                      speed=speed,
                      perception_radius=perception_radius)
    
    for step in range(max_steps):
        swarm.step(strategy)
        if has_converged(swarm.get_positions()):
            return step
    return None #未收敛

# 参数扫描
def sweep():
    results = {}
    if use_comm:
        comm_radii = [10, 20, 30, 50, 80, 120]
        for r in comm_radii:
            print(f"[Comm Mode] Testing communication_radius = {r}")
            t = run_once(communication_radius=r)
            results[r] = t if t is not None else - 1
            print(f" -> Converged at step {results[r]}")
        return comm_radii, results
    
    else:
        print("[No Comm] Running single test")
        t = run_once()
        return [0], {0: t if t is not None else -1}
    
# 绘图
def plot(results_x, results_dict):
    steps = [results_dict[k] for k in results_x]
    steps = [s if s!=-1 else 300 for s in steps] # 用max_steps 替代未收敛

    plt.figure(figsize=(6,4))
    plt.plot(results_x, steps, marker='o')
    if use_comm:
        plt.xlabel("Communication Radius")
    else:
        plt.xlabel("No Communication (single point)")
    plt.ylabel("Convergence Steps")
    plt.title("Swarm Convergence Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize(communication_radius=None, perception_radius=30, num_agents=30, 
              speed=0.5, max_steps=300, strategy="between", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if use_comm:
        swarm = Swarm(num_agents=num_agents, speed=speed, perception_radius=perception_radius, communication_radius=communication_radius)
    else:
        swarm = Swarm(num_agents=num_agents, speed=speed, perception_radius=perception_radius)
    
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=50)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Swarm animation\ncomm_radius={communication_radius}, strategy={strategy}")

    def init():
        scat.set_offsets(np.empty((0,2)))
        return scat,

    def update(frame):
        swarm.step(strategy)
        positions = swarm.get_positions()
        scat.set_offsets(positions)
        if has_converged(positions):
            ax.set_title(f"Converged at step {frame}")
        else:
            ax.set_title(f"Swarm animation\ncomm_radius={communication_radius}, step={frame}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=max_steps, init_func=init, interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    xs, res = sweep()
    plot(xs, res)

    # 选几个典型通信半径，做动画演示
    if use_comm:
        visualize(communication_radius=20)
        visualize(communication_radius=80)
    else:
        visualize()
