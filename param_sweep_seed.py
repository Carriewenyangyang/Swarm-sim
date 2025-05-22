# 并未实现聚类算法
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from q3.swarm import Swarm # 或根据你的路径导入

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
             num_agents=30, speed=0.5, max_steps=300, strategy="between"):
    if seed is not None:
        np.random.seed(seed) # 这里重设随机性

    swarm = Swarm(num_agents=num_agents,
                  speed=speed,
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

# 主sweep函数， 测试通信半径
def sweep_with_repeats(comm_radii, seeds, trials=3, 
                       fix_mode="communication", 
                       fixed_value=30,
                       strategy="between"):
    """
    通用 sweep 函数。
    
    fix_mode: "communication" or "perception"
    - "communication": 固定感知半径，测试通信半径
    - "perception": 固定通信半径，测试感知半径
    fixed_value: 被固定的那个半径的值（如固定感知半径为 30）
    """

    mean_steps = []
    std_steps = []
    mean_clusters = []
    std_clusters = []
    conv_rates = []

    for r in comm_radii:
        all_steps = []
        all_converged = []
        all_clusters = []

        if fix_mode == "communication":
            print(f"Testing communication_radius = {r} (fixed perception_radius = {fixed_value})")
        else:
            print(f"Testing perception_radius = {r} (fixed communication_radius = {fixed_value})")
        #print(f"Testing communication_radius = {r}")
        for seed in seeds:
            for t in range(trials):
                if fix_mode == "communication":
                    step, converged, n_clusters = run_once(
                        seed=seed, 
                        communication_radius=r,
                        perception_radius=fixed_value,
                        strategy=strategy)
                else: # fix_mode == "perception"
                    step, converged, n_clusters = run_once(
                        seed=seed,
                        communication_radius=fixed_value,
                        perception_radius=r,
                        strategy=strategy
                    )

                all_steps.append(step)
                all_converged.append(converged)
                all_clusters.append(n_clusters)

        print(f" -> Mean Steps: {np.mean(all_steps):.2f}, Conv Rate: {np.mean(all_converged):.2f}, Avg Clusters: {np.mean(all_clusters):.2f}")
        mean_steps.append(np.mean(all_steps))
        std_steps.append(np.std(all_steps))
        mean_clusters.append(np.mean(all_clusters))
        std_clusters.append(np.mean(all_clusters))
        conv_rates.append(np.mean(all_converged))

    return mean_steps, std_steps, conv_rates, mean_clusters, std_clusters

# 绘图：误差棒图
def plot_with_errorbars(comm_radii, means, stds):
    plt.figure(figsize=(7,5))
    plt.errorbar(comm_radii, means, yerr=stds, fmt='-o', capsize=5, color='blue')
    plt.xlabel("Communication Radius")
    plt.ylabel("Convergence Steps (lower is better)")
    plt.title("Effect of Communication Radius on Convergence (with error bars)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 绘图：误差棒图，仅针对第三个实验对比规则Behind和between
def plot_cluster_comparison(strategies, avg_cluster_dict, std_clusters_dict):
    """
    strategies: list of strategy names, e.g., ['between', 'behind']
    avg_clusters_dict: dict mapping strategy name to list of avg cluster values (y-axis)
    std_clusters_dict: dict mapping strategy name to list of std cluster values
    """
   
    x = list(range(len(avg_cluster_dict[strategies[0]]))) # 通常是index或对应radius数
    x_labels = ['10', '20', '30', '50', '80', '120']  # 示例横坐标
    colors = plt.cm.tab10.colors  # 提供10种可区分的颜色
    
    plt.figure(figsize=(8,5))

    for i, strategy in enumerate(strategies):
        plt.errorbar(x, 
                     avg_cluster_dict[strategy], 
                     yerr=std_clusters_dict[strategy], 
                     fmt='-o', capsize=5, color=colors[i % len(colors)],
                     label=strategy.capitalize())
    plt.xticks(x, x_labels)
    plt.xlabel('Communication Radius')
    #plt.ylabel('Average Number of Clusters')
    plt.ylabel('Convergence Steps (lower is better)')
    plt.title('Comparison of Clustering: Between vs Behind')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # 参数设置
    # 固定感知半径 = 30，测试不同通信半径：
    
    #communication_radii = [10, 20, 30, 50, 80, 120]
    #seeds = [0, 1, 2, 3, 4]  # 可调整成更多
    #means, stds, conv_rates, clusters = sweep_with_repeats(communication_radii, seeds)
    #plot_with_errorbars(communication_radii, means, stds)

    # 固定通信半径 = 30，测试不同感知半径：
    #perception_radii = [10, 20, 30, 50, 80, 120]
    #seeds = [0, 1, 2, 3, 4]
    #sweep_with_repeats(perception_radii, seeds, fix_mode="perception", fixed_value=30)
    #plot_with_errorbars(communication_radii, means, stds)
    
    # 实验3，比较Behind和between策略
    strategies = ["between", "behind"]
    perception_radii = [10, 20, 30, 50, 80, 120]  # 只跑一个感知半径值，作为固定条件
    avg_clusters_dict = {}
    std_clusters_dict = {}

    for strategy in strategies:
        print(f"Running strategy: {strategy}")
        
        seeds = [0, 1, 2, 3, 4]
        mean_steps, std_steps, conv_rates, mean_clusters, std_clusters = sweep_with_repeats(
            perception_radii,
            seeds,
            fix_mode="perception",     # 选择变化感知半径，但一个值也是固定
            fixed_value=50,            # 通信半径固定 = 50
            strategy=strategy          # 关键点：传入行为策略
        )

        avg_clusters_dict[strategy] = mean_steps #mean_clusters
        std_clusters_dict[strategy] = std_steps #std_clusters
    plot_cluster_comparison(strategies, avg_clusters_dict, std_clusters_dict)
