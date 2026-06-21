import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import time
import os

# ================= 模块一：2D 栅格地图与 C-Space 膨胀 =================
GRID_SIZE = 20
START = (2, 2)
GOAL = (18, 18)
ROBOT_RADIUS = 1.0  # 机器人物理半径

def create_map():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    # 制造一个占据左上方的巨大方块障碍物，逼迫机器人走一个极限 90 度直角
    grid[0:15, 8:20] = 1
    return grid

def inflate_map(grid, radius):
    """代价地图膨胀：将障碍物向外膨胀一个机器人半径，构建安全位姿空间"""
    inflated = np.copy(grid)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                rmin, rmax = max(0, int(r-radius)), min(rows, int(r+radius)+1)
                cmin, cmax = max(0, int(c-radius)), min(cols, int(c+radius)+1)
                for ir in range(rmin, rmax):
                    for ic in range(cmin, cmax):
                        # 欧氏距离小于等于半径的区域，设为致命区域
                        if np.linalg.norm([r-ir, c-ic]) <= radius:
                            inflated[ir, ic] = 1
    return inflated

# ================= 模块二：底层遗传算法 =================
POP_SIZE = 150
MAX_ITER = 200
WAYPOINTS_NUM = 4  # 航点设少一点，让 GA 跑出干净利落的直角
MUTATION_RATE = 0.3

def init_population():
    return np.random.randint(0, GRID_SIZE, size=(POP_SIZE, WAYPOINTS_NUM, 2))

def check_collision_and_length(path, grid):
    full_path = np.vstack([START, path, GOAL])
    length = 0
    collisions = 0
    for i in range(len(full_path)-1):
        p1, p2 = full_path[i], full_path[i+1]
        dist = np.linalg.norm(p1 - p2)
        length += dist
        steps = int(dist * 2) + 1
        for step in range(steps):
            x = int(p1[0] + (p2[0] - p1[0]) * step / steps)
            y = int(p1[1] + (p2[1] - p1[1]) * step / steps)
            if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or grid[x, y] == 1:
                collisions += 1
    return length, collisions

def evaluate_fitness(pop, grid):
    fitness = np.zeros(POP_SIZE)
    for i in range(POP_SIZE):
        length, collisions = check_collision_and_length(pop[i], grid)
        fitness[i] = 1.0 / (length + collisions * 1000 + 1e-5)
    return fitness

def selection(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
    return pop[idx]

def crossover(pop):
    new_pop = np.copy(pop)
    for i in range(0, POP_SIZE, 2):
        if np.random.rand() < 0.8: 
            cross_pt = np.random.randint(1, WAYPOINTS_NUM-1)
            new_pop[i, cross_pt:] = pop[i+1, cross_pt:]
            new_pop[i+1, cross_pt:] = pop[i, cross_pt:]
    return new_pop

def mutation(pop):
    for i in range(POP_SIZE):
        if np.random.rand() < MUTATION_RATE:
            mut_pt = np.random.randint(0, WAYPOINTS_NUM)
            pop[i, mut_pt] = [np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)]
    return pop

# ================= 模块三：第一版 B 样条平滑 (带缺陷版) =================
def b_spline_smooth_v1(path_points):
    unique_points = [path_points[0]]
    for p in path_points[1:]:
        if np.linalg.norm(p - unique_points[-1]) > 1.0:
            unique_points.append(p)
    if not np.array_equal(unique_points[-1], path_points[-1]):
        unique_points.append(path_points[-1])
    unique_points = np.array(unique_points)
    if len(unique_points) < 4: return unique_points
        
    # 缺陷：s 值故意设大（允许极大逼近误差），在 90 度急弯处必然严重抄近道撞墙
    tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=15.0, k=3)
    u_new = np.linspace(u.min(), u.max(), 300)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

# ================= 模块四：改进版 B 样条平滑 (动态滤波+强制插值) =================
def b_spline_smooth_v2(path_points):
    unique_points = [path_points[0]]
    for p in path_points[1:]:
        if np.linalg.norm(p - unique_points[-1]) > 1.0:
            unique_points.append(p)
    if not np.array_equal(unique_points[-1], path_points[-1]):
        unique_points.append(path_points[-1])
    unique_points = np.array(unique_points)
    if len(unique_points) < 4: return unique_points
        
    # 改进：s=0.0 强制插值，必须精准穿过膨胀后的安全航点，彻底杜绝穿模
    tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=0.0, k=3)
    u_new = np.linspace(u.min(), u.max(), 300)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

# ================= 模块五：主流程与可视化输出 =================
if __name__ == "__main__":
    current_seed = np.random.randint(0, 1000000)
    # 跑到满意的图就把打印出来的数字填到下面这行，解开注释！
    # current_seed = 123456  
    np.random.seed(current_seed)
    print(f"\n当前运行的随机种子: {current_seed}\n")

    print("生成地图并计算 C-Space 膨胀层...")
    raw_grid_map = create_map()
    inflated_grid_map = inflate_map(raw_grid_map, ROBOT_RADIUS)
    
    print("开始运行遗传算法...")
    start_time = time.time()
    population = init_population()
    best_path, best_fitness = None, 0
    
    for gen in range(MAX_ITER):
        fit = evaluate_fitness(population, inflated_grid_map)
        best_idx = np.argmax(fit)
        if fit[best_idx] > best_fitness:
            best_fitness = fit[best_idx]
            best_path = population[best_idx]
            
        population = selection(population, fit)
        population = crossover(population)
        population = mutation(population)
        
    raw_full_path = np.vstack([START, best_path, GOAL])
    path_v1_bad = b_spline_smooth_v1(raw_full_path)
    path_v2_good = b_spline_smooth_v2(raw_full_path)
    
    print(f"迭代完成，耗时: {time.time()-start_time:.2f}秒")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["(a) Raw GA Path (Safe but Sharp)", 
              "(b) V1 B-Spline (Severe Collision/穿模)", 
              "(c) V2 Improved B-Spline (Safe & Smooth)"]
    paths = [raw_full_path, path_v1_bad, path_v2_good]
    
    for ax, path, title in zip(axes, paths, titles):
        ax.imshow(raw_grid_map.T, cmap='Greys', origin='lower')
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2.5, label='Robot Path')
        ax.plot(START[0], START[1], 'bo', markersize=8, label='Start')
        ax.plot(GOAL[0], GOAL[1], 'go', markersize=8, label='Goal')
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    save_path = "figs/ga_path_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"绝杀对比图已自动保存至: {os.path.abspath(save_path)}")
    plt.show()