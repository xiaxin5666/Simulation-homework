import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ===================== 1. 设置基础参数 =====================
lambda_exp = 1  # 指数分布的参数λ
pop_mean = 1 / lambda_exp  # 总体均值
pop_var = 1 / (lambda_exp **2)  # 总体方差
n_sample = 50  # 每次抽样的样本量（满足大样本要求）
n_repeat = 10000  # 重复抽样的次数（次数越多，分布越清晰）

# ===================== 2. 生成数据 =====================
# 生成指数分布的总体（用于对比）
pop_data = np.random.exponential(scale=1/lambda_exp, size=100000)

# 模拟多次抽样，计算每次的样本均值
sample_means = []
for _ in range(n_repeat):
    # 每次抽n_sample个样本（独立同分布）
    sample = np.random.exponential(scale=1/lambda_exp, size=n_sample)
    # 计算样本均值并保存
    sample_means.append(np.mean(sample))
sample_means = np.array(sample_means)

# ===================== 3. 计算理论正态分布参数 =====================
# 中心极限定理预测的均值和标准差
clt_mean = pop_mean
clt_std = np.sqrt(pop_var / n_sample)  # 样本均值的标准差=总体标准差/√n

# ===================== 4. 可视化对比 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：指数分布总体的分布（严重右偏）
ax1.hist(pop_data, bins=50, density=True, alpha=0.7, color='lightcoral', label='指数分布总体')
# 绘制指数分布的理论曲线
x_exp = np.linspace(0, 8, 100)
ax1.plot(x_exp, stats.expon.pdf(x_exp, scale=1/lambda_exp), 'r-', linewidth=2, label='指数分布理论曲线')
ax1.set_title('指数分布总体的分布（λ=1）')
ax1.set_xlabel('数值')
ax1.set_ylabel('概率密度')
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：样本均值的分布（趋近正态）
ax2.hist(sample_means, bins=50, density=True, alpha=0.7, color='lightblue', label='样本均值分布')
# 绘制CLT预测的正态分布曲线
x_norm = np.linspace(clt_mean - 3*clt_std, clt_mean + 3*clt_std, 100)
ax2.plot(x_norm, stats.norm.pdf(x_norm, loc=clt_mean, scale=clt_std), 'b-', linewidth=2, label='CLT预测正态分布')
ax2.set_title(f'样本均值的分布（n={n_sample}）')
ax2.set_xlabel('样本均值')
ax2.set_ylabel('概率密度')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===================== 5. 输出关键统计量验证 =====================
print("=== 验证中心极限定理 ===")
print(f"总体均值：{pop_mean:.2f} | 样本均值的均值：{np.mean(sample_means):.2f}")
print(f"理论样本均值标准差：{clt_std:.4f} | 实际样本均值标准差：{np.std(sample_means, ddof=1):.4f}")