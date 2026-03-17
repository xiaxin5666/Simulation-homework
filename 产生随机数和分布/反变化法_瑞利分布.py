import numpy as np
import matplotlib.pyplot as plt

# 1. 设置参数
sigma = 1.0
num_samples = 100000

# 2. 生成均匀分布随机数 u ~ U(0, 1)
u = np.random.uniform(0, 1, num_samples)

# 3. 应用反变换公式得到瑞利分布随机变量 r
r = sigma * np.sqrt(-2 * np.log(u))

# 4. 准备理论 PDF 曲线数据
x = np.linspace(0, 5, 1000)
pdf_theory = (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2))

# 5. 可视化结果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))

# 绘制实验统计的直方图 (density=True 保证面积之和为1，以便与PDF比较)
plt.hist(r, bins=100, density=True, alpha=0.6, color='skyblue', label='Experimental Statistics')

# 绘制理论 PDF 曲线
plt.plot(x, pdf_theory, 'r-', lw=2, label='Theoretical PDF')

plt.title(f'(瑞利分布, sigma={sigma})')
plt.xlabel('Value (r)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('瑞利分布.png')
plt.show()