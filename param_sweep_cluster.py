import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.animation as animation
from q3.swarm import Swarm #确保导入的是你第二份或第一份的Swarm，看你用哪个

# 判断是否收敛(所有agent距离中心小于某个阈值)
def has_converged(positions, thresh=5.0):
    center = positions.mean(axis=0)
    maxd = np.max(np.linalg.norm(positions - center, axis=1))
    return maxd < thresh

def get_cluster_count(positions, eps=5.0, min_samples=2):
    #from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters

# 单次实验
def run_once(perception_radius, speed=0.5, num_agents=30, 
             max_steps=300, strategy="between", seed=None):
    if seed is not None:
        np.random.seed(seed) #固定随机性
    
    swarm = Swarm(num_agents=num_agents, speed=speed, perception_radius=perception_radius)
    for step in range(max_steps):
        swarm.step(strategy)
    
    final_positions = swarm.get_positions()
    converged = has_converged(final_positions, thresh=5.0)
    n_clusters = get_cluster_count(final_positions, eps=0.6) # eps和thresh接近
    return converged, n_clusters

# 参数扫描
def param_sweep(seed=42):
    perception_radii = [10, 20, 30, 40, 50, 60, 80, 100]
    results_converged = []
    results_clusters = []

    for r in perception_radii:
        print(f"Running perception_radius = {r} ...")
        converged, n_clusters = run_once(perception_radius=r, seed=seed)
        results_converged.append(int(converged))
        results_clusters.append(n_clusters)
        print(f"-> Converged: {converged}, Clusters: {n_clusters}")
    
    return perception_radii, results_converged, results_clusters

# 可视化结果
def plot_results(perception_radii, converged_list, cluster_list):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel("Perception Radius")
    ax1.set_ylabel("Converged (1=True, 0=False)", color=color)
    ax1.plot(perception_radii, converged_list, marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() #第二个Y轴
    color = 'tab:red'
    ax2.set_ylabel("Number of Clusters", color=color)
    ax2.plot(perception_radii, cluster_list, marker='s', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Effect of Perception radius on Convergence and Clustering")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

# 感兴趣时，输出动画版的结果
def visualize_swarm(perception_radius, speed=0.5, num_agents=30, 
                    strategy="between", seed=None, save_path=None):
    if seed is not None:
        np.random.seed(seed)
    swarm = Swarm(num_agents=num_agents, speed=speed, 
                  perception_radius=perception_radius)
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=50)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Swarm Visualization: radius={perception_radius}, strategy={strategy}")

    def init():
        scat.set_offsets(np.empty((0,2)))
        return scat, # 注意返回是元组
    
    def update(frame):
        swarm.step(strategy)
        positions = swarm.get_positions()
        scat.set_offsets(positions)

            # 判断是否收敛
        if has_converged(positions, thresh=5.0):
            ax.set_title(f"Converged at step {frame}")
        else:
            ax.set_title(f"Swarm Strategy: {strategy} (Step {frame})")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=100, init_func=init,
                                   interval=100, blit=True)
    # 保存为GIF（可选）
    if save_path:
        print(f"Saving GIF to {save_path} ...")
        ani.save(save_path, writer='pillow', fps=10)

    # 添加这一行：把 ani 存起来
    plt.ani = ani  # 关键点！
    plt.show()

if __name__ == "__main__":
    pr, convs, clusts = param_sweep()
    plot_results(pr, convs, clusts)
     # 可视化两个典型状态
    print("Visualizing for perception_radius = 10 (likely not converged)...")
    visualize_swarm(perception_radius=100, seed=42, save_path="q3/swarm_animation.gif")

    print("Visualizing for perception_radius = 20 (likely converged)...")
    visualize_swarm(perception_radius=20, seed=42)