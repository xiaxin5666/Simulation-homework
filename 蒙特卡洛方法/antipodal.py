import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

N = np.array([1000, 10000, 100000, 1000000])
EbN0_indB = np.arange(-2, 10, 2)  # Python range is [start, end), so use 10 to include 8
Ebn0 = 10 ** (EbN0_indB / 10.0)
E = 1
times = 50
err_rate = np.zeros((len(Ebn0), len(N), times))
print("开始仿真计算...")
for i_N in range(len(N)):
    for i_snr in range(len(Ebn0)):  # 对应 iii
        # 计算噪声标准差
        sigma = E / np.sqrt(2 * Ebn0[i_snr])
        for i_t in range(times):  # 对应 ii
            # 生成随机比特 (0 或 1)
            r = np.random.rand(N[i_N])
            source = r > 0.5
            # BPSK 调制 (0 -> 1, 1 -> -1)
            x = 1 - 2 * source
            # 加入高斯白噪声
            noise = np.random.randn(N[i_N]) * sigma
            # 接收信号
            y = E * x + noise
            # 判决 (y<=0 -> 1, y>0 -> 0)
            result = y <= 0
            # 统计误码
            error_num = np.sum(result != source)
            # 存储误码率 (注意Python索引顺序)
            err_rate[i_snr, i_N, i_t] = error_num / N[i_N]
# ---------------------------------------------------------
# 理论值计算与第一次绘图
# ---------------------------------------------------------
# 理论误码率
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))
plt.figure(0)
ber_check = err_rate[:, 3, 0]  # 索引3对应第4个N (100000)
plt.semilogy(EbN0_indB, ber_check, 'r', label='Simulation',marker='o')
plt.semilogy(EbN0_indB, pe_theory, 'g', label='Theory',marker='^')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.legend()
plt.title("BER")
plt.xlabel('Eb/N0(dB)', fontsize=14)
plt.ylabel('BER', fontsize=14)
# ---------------------------------------------------------
# 相对误差方差计算
# ---------------------------------------------------------
variance = np.zeros((len(Ebn0), len(N)))
for i_snr in range(len(Ebn0)):
    for i_N in range(len(N)):
        # 提取50次实验的数据
        ber_samples = err_rate[i_snr, i_N, :]
        # 计算相对误差
        relative_error = (ber_samples - pe_theory[i_snr]) / pe_theory[i_snr]
        # 计算方差
        variance[i_snr, i_N] = np.var(relative_error, ddof=1)
# ---------------------------------------------------------
# 最终绘图 Figure 1
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))  # 尺寸大致对应 PaperSize
# 设置网格和坐标轴属性
ax.set_yscale('log')
ax.grid(True, which='both', linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
# 标记样式列表 (对应 MATLAB set semilogy1(1)... 的标记)
markers = ['o', '^', 's', 'x', 'd']  # s=square, d=diamond
# 标签列表
labels = ['1000bits', '10000bits', '100000bits', '1000000bits']
# 绘制多条曲线
for i in range(len(N)):
    ax.semilogy(EbN0_indB, variance[:, i],
                color='k',  # Color=[0 0 0]
                linewidth=2,  # LineWidth',2
                marker=markers[i],  # 对应不同的Marker
                markersize=8,
                label=labels[i])  # DisplayName
# 设置标签
plt.xlabel('Eb/N0(dB)', fontsize=14)
plt.ylabel('Variance', fontsize=14)
# 设置图例
plt.legend(loc='best', fontsize=10)
print("绘图完成。")
plt.show()