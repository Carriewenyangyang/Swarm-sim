# 并未实现聚类算法
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from q3.swarm import Swarm # 或根据你的路径导入
import itertools
import os

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

# 跑一次实验(指定种子)，测试感知半径
def run_once(seed=None, perception_radius=30, communication_radius=50,
             num_agents=30, speed=0.5, max_steps=300, strategy="between",
             hetero_speed=False):
    if seed is not None:
        np.random.seed(seed) # 这里重设随机性

    if hetero_speed:
        speed_list = np.random.uniform(0.3, 0.7, size=num_agents)
    else:
        speed_list = [speed] * num_agents

    swarm = Swarm(num_agents=num_agents,
                  speed_list=speed_list,
                  perception_radius=perception_radius,
                  communication_radius=communication_radius) # 所有实验用的是相同的初始位置
    
    for step in range(max_steps):
        swarm.step(strategy)
        if has_converged(swarm.get_positions()):
            converged= True
            break
    else:
        step = max_steps
        converged = False

    final_positions = swarm.get_positions()
    n_clusters = get_cluster_count(final_positions, eps=0.6) # eps和thresh接近
    return step, converged, n_clusters
    #return max_steps # 若未收敛，返回最大步数

# 主sweep函数， 测试速度异质性
def test_heterogeneous_speed(seeds, strategy="between", trials=3):
    modes = [False, True]  # False: 同质，True: 异质
    labels = ["Homogeneous Speed", "Heterogeneous Speed"]

    results = {}

    for hetero_speed, label in zip(modes, labels):
        print(f"\nRunning: {label}")
        all_steps = []
        all_converged = []
        all_clusters = []

        for seed in seeds:
            for _ in range(trials):
                step, converged, n_clusters = run_once(
                    seed=seed,
                    perception_radius=50,
                    communication_radius=50,
                    strategy=strategy,
                    hetero_speed=hetero_speed
                )
                all_steps.append(step)
                all_converged.append(converged)
                all_clusters.append(n_clusters)

        results[label] = {
            "mean_steps": np.mean(all_steps),
            "std_steps": np.std(all_steps),
            "conv_rate": np.mean(all_converged),
            "avg_clusters": np.mean(all_clusters),
        }

    return results

def plot_hetero_results(results):
    labels = list(results.keys())
    mean_steps = [results[l]["mean_steps"] for l in labels]
    conv_rates = [results[l]["conv_rate"] for l in labels]
    avg_clusters = [results[l]["avg_clusters"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].bar(x, mean_steps, width)
    ax[0].set_title("Mean Steps")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=15)

    ax[1].bar(x, conv_rates, width)
    ax[1].set_title("Convergence Rate")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=15)

    ax[2].bar(x, avg_clusters, width)
    ax[2].set_title("Average Clusters")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=15)

    plt.tight_layout()
    plt.show()


# 感兴趣时，输出动画版的结果
def visualize_param_combinations(perception_list, comm_list, speed_list,
                                  num_agents=30, strategy="between", 
                                  output_dir="swarm_gifs", seed=42):

    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

    # 所有参数组合
    param_combinations = list(itertools.product(perception_list, comm_list, speed_list))

    for i, (perception_radius, communication_radius, speed) in enumerate(param_combinations):
        print(f"Generating GIF {i+1}/{len(param_combinations)}: "
              f"perception={perception_radius}, communication={communication_radius}, speed={speed}")

        # 初始化 swarm
        np.random.seed(seed)  # 保证相同初始状态
        swarm = Swarm(num_agents=num_agents, speed=speed, 
                      perception_radius=perception_radius, 
                      communication_radius=communication_radius)

        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=50)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        def init():
            scat.set_offsets(np.empty((0, 2)))
            return scat,

        def update(frame):
            swarm.step(strategy)
            positions = swarm.get_positions()
            scat.set_offsets(positions)

            if has_converged(positions, thresh=5.0):
                ax.set_title(f"Converged at step {frame}")
            else:
                ax.set_title(f"Step {frame}")
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=100, init_func=init,
                                      interval=100, blit=True)

        filename = f"swarm_p{perception_radius}_c{communication_radius}_s{speed:.2f}.gif"
        filepath = os.path.join(output_dir, filename)
        ani.save(filepath, writer='pillow', fps=10)
        plt.close(fig)  # 防止显示太多图

        print(f"Saved: {filepath}")

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    results = test_heterogeneous_speed(seeds)

    print("\n=== Speed Heterogeneity Comparison Results ===")
    for mode, res in results.items():
        print(f"\n[{mode}]")
        print(f"Mean Steps: {res['mean_steps']:.2f}")
        print(f"Conv Rate: {res['conv_rate']:.2f}")
        print(f"Avg Clusters: {res['avg_clusters']:.2f}")
    plot_hetero_results(results)

    perception_list = [50]
    comm_list = [50]
    speed_list = [1.0]

    visualize_param_combinations(perception_list, comm_list, speed_list)

