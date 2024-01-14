import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

from tqdm import tqdm

def sphere_func(x):
    return np.sum(x ** 2)


def schwefel_2_22_func(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def rosenbrock_func(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def schwefel_func(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin_func(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley_func(x):
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.exp(1)

# Define the Velocity Limit Handling Algorithm
def Velocity_Limit_Handling(d, v, vl, f):
    for i in range(d):
        if (v[i] > vl) or (v[i] < -vl):
            if f >= 0.5:
                v[i] = min(vl, max(-vl, v[i]))
            else:
                v[i] = np.random.rand() * 2 * vl - vl
    return v


# Define the Position Limit Handling Algorithm
def Position_Limit_Handling(d, xi, xmax, xmin):
    for i in range(d):
        if xi[i] > xmax or xi[i] < xmin:
            xi[i] = np.random.rand() * (xmax - xmin) - xmin
    return xi



def PSO_SAVL(func):
    # 参数设置及初始化
    N = 20  # 种群数目
    D = 50  # 维度
    T = 1000  # 最大迭代次数SS
    c1 = c2 = 2.05  # 个体学习因子与群体学习因子
    w_max = 0.9  # 权重系数最大值
    w_min = 0.4  # 权重系数最小值
    μ_max = 0.7  #
    μ_min = 0.4  #
    alpha = (1 / μ_min) - 1
    beta = -math.log(((1 / μ_max) - 1) / alpha)
    opt_func = func
    x_max = np.ones(D) * 32  # 每个维度最大取值范围
    x_min = np.ones(D) * -32  # 每个维度最小取值范围
    v_max = np.ones(D) * 3.2  # 每个维度粒子的最大速度
    v_min = np.ones(D) * -3.2  # 每个维度粒子的最小速度

    vl = μ_max * x_max  # 速度的限制边界

    # 种群个体及速度初始化I
    x = np.random.rand(N, D) * (x_max - x_min) + x_min  # 初始化每个粒子的位置 x为数组
    v = np.random.rand(N, D) * (v_max - v_min) + v_min  # 初始化每个粒子的速度 v为数组

    # 个体最优和群体最有初始化
    p = x  # 用来存储每一个粒子的历史最优位置
    p_best = np.ones((N, 1))  # 每行存储的是最优值
    for i in range(N):  # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
        p_best[i] = opt_func(x[i, :])

    g_best = opt_func(x[0, :])  # 设置真的全局最优值
    gb = np.ones(T)  # 用于记录每一次迭代的全局最优值
    x_best = np.ones(D)  # 用于存储最优粒子的位置

    Euc_distance = np.zeros((N, N))  # 创建一个用于存储欧氏距离的二维数组
    mean_distance = np.zeros(N)  # 创建一个用于存储平均距离的一维数组

    # 按照公式依次迭代直到满足精度或者迭代次数
    for t in range(T):
        w = w_max - (w_max - w_min) * t / T
        for i in range(N):
            for k in range(N):
                Euc_distance[i][k] = np.linalg.norm(v[i, :] - v[k, :])
            mean_distance[i] = np.sum(Euc_distance[i, :]) / (N - 1)
        f = (mean_distance[np.argmax(p_best)] - np.min(mean_distance)) / (np.max(mean_distance) - np.min(mean_distance))
        vl = (1 / (1 + alpha * math.exp(-beta * f))) * x_max

        for j in range(N):
            v[j, :] = w * v[j, :] + c1 * np.random.rand(D) * (p[j, :] - x[j, :]) + c2 * np.random.rand(D) * (
                    x_best - x[j, :])
            is_within_vrange = np.all((v[j, :] >= -vl) & (v[j, :] <= vl))
            if not is_within_vrange:
                v[j, :] = Velocity_Limit_Handling(d=D, v=v[j, :], vl=vl[j], f=f)

            x[j, :] = x[j, :] + v[j, :]
            is_within_prange = np.all((x[j, :] >= x_min) & (x[j, :] <= x_max))
            if not is_within_prange:
                x[j, :] = Position_Limit_Handling(d=D,xi= x[j, :],xmax=x_max[j],xmin=x_min[j])
            # 更新个体最优值和全局最优值

            if p_best[j] > opt_func(x[j, :]):
                p_best[j] = opt_func(x[j, :])
                p[j, :] = x[j, :].copy()
            # 更新全局最优值
            if g_best > p_best[j]:
                g_best = p_best[j]
                x_best = x[j, :].copy()
        gb[t] = g_best
    return gb
def save_results(gb, optimization_method, optimization_function):
    filename = f"result_{optimization_method}_{optimization_function}.txt"
    np.savetxt(filename, gb, fmt="%f")
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    gb = np.zeros((30,1000))
    success_times = 0
    for i in tqdm(range(30), desc="PSO Iterations", unit="iteration"):  # 使用tqdm
        gb_temp = PSO_SAVL(schwefel_2_22_func)
        gb[i,:] =gb_temp
    gb_std = np.var(gb[:,999])
    print(gb_std)
    # save_results(gb, "PSO_SAVL", "ackley_func")
    # matplotlib.rc("font", family="KaiTi")
    # matplotlib.rcParams["axes.unicode_minus"] = False
    # plt.plot(range(1000), gb)
    # plt.xlabel("迭代次数")
    # plt.ylabel("适应度值")
    # plt.title("适应度进化曲线")
    # plt.show()