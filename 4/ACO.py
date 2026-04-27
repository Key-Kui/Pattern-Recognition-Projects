import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ================= 1. 环境配置 =================
os.makedirs('Optimization/figs', exist_ok=True)
plt.rcParams['figure.figsize'] = (10, 8)
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("【蚁群算法 ACO】中国34城市物流路径规划 (TSP)")
print("=" * 60)

# ================= 2. 内置真实数据集 (中国34个省级行政区坐标) =================
# 经纬度近似坐标 (方便直接运行，无需下载CSV)
cities = {
    '北京': (116.40, 39.90), '天津': (117.20, 39.12), '上海': (121.47, 31.23), '重庆': (106.55, 29.56),
    '哈尔滨': (126.63, 45.75), '长春': (125.32, 43.90), '沈阳': (123.43, 41.80), '呼和浩特': (111.75, 40.84),
    '石家庄': (114.48, 38.03), '太原': (112.53, 37.87), '西安': (108.95, 34.27), '济南': (117.00, 36.65),
    '郑州': (113.62, 34.75), '合肥': (117.27, 31.86), '南京': (118.78, 32.04), '杭州': (120.19, 30.26),
    '南昌': (115.89, 28.68), '福州': (119.30, 26.08), '武汉': (114.31, 30.52), '长沙': (112.93, 28.23),
    '广州': (113.23, 23.16), '南宁': (108.33, 22.84), '海口': (110.35, 20.02), '成都': (104.06, 30.67),
    '贵阳': (106.71, 26.57), '昆明': (102.73, 25.04), '拉萨': (91.11, 29.65), '兰州': (103.82, 36.06),
    '西宁': (101.78, 36.62), '银川': (106.27, 38.47), '乌鲁木齐': (87.68, 43.77), '台北': (121.50, 25.03),
    '香港': (114.17, 22.28), '澳门': (113.54, 22.19)
}

city_names = list(cities.keys())
coords = np.array(list(cities.values()))
n_cities = len(city_names)

# 计算距离矩阵 (欧氏距离模拟)
dist_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        else:
            dist_matrix[i, j] = 1e-10 # 避免除零

# ================= 3. 蚁群算法核心逻辑 (ACO) =================
class AntColonyOptimization:
    def __init__(self, n_ants=80, n_iter=640, alpha=1.0, beta=5.0, rho=0.1, Q=100):
        self.n_ants = n_ants      # 蚂蚁数量
        self.n_iter = n_iter      # 迭代次数
        self.alpha = alpha        # 信息素重要程度
        self.beta = beta          # 启发式因子重要程度
        self.rho = rho            # 信息素挥发系数
        self.Q = Q                # 信息素常数
        
    def fit(self, dist_matrix):
        n = dist_matrix.shape[0]
        # 初始化信息素矩阵和启发式信息 (距离的倒数)
        pheromone = np.ones((n, n))
        eta = 1.0 / dist_matrix
        
        best_path = None
        best_dist = float('inf')
        self.history_best_dist = []
        
        print("开始蚁群搜索...")
        for iter_idx in range(self.n_iter):
            paths = []
            path_dists = []
            
            # 1. 蚂蚁寻路阶段
            for ant in range(self.n_ants):
                # 随机选择一个起点
                current_city = np.random.randint(n)
                unvisited = list(range(n))
                unvisited.remove(current_city)
                path = [current_city]
                dist = 0.0
                
                while unvisited:
                    # 计算当前城市到未访问城市的状态转移概率
                    probs = np.zeros(len(unvisited))
                    for i, next_city in enumerate(unvisited):
                        probs[i] = (pheromone[current_city, next_city] ** self.alpha) * \
                                   (eta[current_city, next_city] ** self.beta)
                    probs = probs / probs.sum()
                    
                    # 轮盘赌选择下一个城市
                    next_city = np.random.choice(unvisited, p=probs)
                    path.append(next_city)
                    dist += dist_matrix[current_city, next_city]
                    unvisited.remove(next_city)
                    current_city = next_city
                
                # 回到起点，形成闭环
                dist += dist_matrix[path[-1], path[0]]
                paths.append(path)
                path_dists.append(dist)
                
                # 记录全局最优
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
            
            self.history_best_dist.append(best_dist)
            
            # 2. 信息素更新阶段 (挥发 + 释放)
            pheromone = (1 - self.rho) * pheromone
            for path, dist in zip(paths, path_dists):
                for i in range(n - 1):
                    pheromone[path[i], path[i+1]] += self.Q / dist
                    pheromone[path[i+1], path[i]] += self.Q / dist
                # 闭环信息素
                pheromone[path[-1], path[0]] += self.Q / dist
                pheromone[path[0], path[-1]] += self.Q / dist
                
            if (iter_idx + 1) % 50 == 0:
                print(f"  -> 迭代 {iter_idx+1}/{self.n_iter}, 当前最短距离: {best_dist:.2f}")
                
        self.best_path = best_path
        self.best_dist = best_dist
        return self

aco = AntColonyOptimization(n_ants=80, n_iter=200)
aco.fit(dist_matrix)

# ================= 4. 生成可视化图表 =================
print("\n生成图表中...")

# 图 1: ACO 收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, aco.n_iter + 1), aco.history_best_dist, linewidth=2, color='#2980b9')
plt.title('蚁群算法 (ACO) 求解 TSP 收敛曲线', fontsize=14)
plt.xlabel('迭代次数 (Iterations)')
plt.ylabel('最优路径总距离 (Best Distance)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('Optimization/figs/ACO_Convergence.png', dpi=150, bbox_inches='tight')
plt.close()

# 图 2: 最优路径地图可视化
plt.figure(figsize=(12, 10))
# 画城市点
plt.scatter(coords[:, 0], coords[:, 1], c='#e74c3c', s=60, zorder=5)

# 标出城市名称
for i, name in enumerate(city_names):
    plt.text(coords[i, 0] + 0.3, coords[i, 1] - 0.2, name, fontsize=9, zorder=6)

# 画出最优连接线 (闭环)
path = aco.best_path
for i in range(n_cities):
    start_city = path[i]
    end_city = path[(i + 1) % n_cities] # 回到起点
    plt.plot([coords[start_city, 0], coords[end_city, 0]], 
             [coords[start_city, 1], coords[end_city, 1]], 
             c='#27ae60', linewidth=1.5, alpha=0.8, zorder=3)

# 标出起点 (用蓝色大星号表示全国物流中心)
start_idx = path[0]
plt.scatter(coords[start_idx, 0], coords[start_idx, 1], c='blue', marker='*', s=300, zorder=10, label='Logistics Hub (起点)')

plt.title(f'全国34个城市物流最优配送路径\nTotal Distance: {aco.best_dist:.2f}', fontsize=16)
plt.xlabel('经度 (Longitude)')
plt.ylabel('纬度 (Latitude)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('Optimization/figs/ACO_Optimal_Route.png', dpi=150, bbox_inches='tight')
plt.close()

print("运行完毕！全国路线图与收敛曲线已保存至 Optimization/figs 文件夹。")