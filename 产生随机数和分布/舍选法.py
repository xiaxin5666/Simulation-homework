import numpy as np
import time

def f(x):
    return 20 * x * (1 - x)**3

M = 2.109375  # f(x) 的最大值
num_needed = 100000

# --- 方式 A: 矩形舍选 ---
start_a = time.time()
accepted_a = []
total_tried_a = 0

while len(accepted_a) < num_needed:
    total_tried_a += 1
    x_cand = np.random.uniform(0, 1)
    u = np.random.uniform(0, M)
    if u <= f(x_cand):
        accepted_a.append(x_cand)

end_a = time.time()
efficiency_a = num_needed / total_tried_a

# --- 方式 B: 简单优化 (使用分段包络，例如在 [0,1] 上用更紧凑的常数) ---
# 这里演示原理：将 [0,1] 分为 [0, 0.5] 和 [0.5, 1]，分别取局部最大值
m1, m2 = f(0.25), f(0.5) # 分段包络高度
start_b = time.time()
accepted_b = []
total_tried_b = 0

while len(accepted_b) < num_needed:
    total_tried_b += 2 # 简化逻辑
    # 随机选一个区间，按区间面积比例采样（此处为示意简化版）
    # 实际优化应使用紧致的分布函数进行抽样
    x_cand = np.random.uniform(0, 1)
    limit = m1 if x_cand < 0.5 else m2
    u = np.random.uniform(0, limit)
    if u <= f(x_cand):
        accepted_b.append(x_cand)

end_b = time.time()
efficiency_b = num_needed / (total_tried_b/2) # 修正计数

print(f"矩形舍选效率: {efficiency_a:.2%}, 耗时: {end_a-start_a:.4f}s")
print(f"优化舍选效率: {efficiency_b:.2%}, 耗时: {end_b-start_b:.4f}s")