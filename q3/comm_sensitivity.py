# comm_sensitivity.py

import numpy as np
import matplotlib.pyplot as plt
from swarm import Swarm

def has_converged(positions, thresh=5.0):
    """判断一组位置是否收敛：所有 agent 到质心的最大距离 < thresh。"""
    center = positions.mean(axis=0)
    maxd = np.max(np.linalg.norm(positions - center, axis=1))
    return maxd < thresh

def run_once(comm_radius, 
             num_agents=30, speed=0.6, perception_radius=30, 
             max_steps=500, strategy="between"):
    swarm = Swarm(num_agents, speed=speed,
                  perception_radius=perception_radius,
                  communication_radius=comm_radius)
    for step in range(1, max_steps+1):
        swarm.step(strategy)
        if has_converged(swarm.get_positions()):
            return step
    return None

def batch_test(comm_radii):
    results = {}
    for r in comm_radii:
        print(f"Testing communication_radius = {r} ...")
        t = run_once(r)
        results[r] = t if t is not None else max_steps
        print(f" → Converged at step {results[r]}")
    return results

if __name__ == "__main__":
    comm_radii = [10, 20, 30, 50, 80, 120]
    max_steps = 500

    # 批量测试
    results = {}
    for r in comm_radii:
        t = run_once(r, max_steps=max_steps)
        results[r] = t if t is not None else max_steps

    # 绘制折线图
    plt.figure(figsize=(6,4))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel("Communication Radius")
    plt.ylabel("Convergence Time Steps")
    plt.title("Effect of Communication Radius on Swarm Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
