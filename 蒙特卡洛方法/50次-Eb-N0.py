import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ========== 基础配置：解决负号显示 + 字体 ==========
plt.rcParams['font.sans-serif'] = ['SimHei','SimHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 仿真参数（聚焦N=10000，50次实验） ==========
N = np.array([1000, 10000, 100000, 1000000])
target_N_idx = 1  # 对应10000比特
EbN0_indB = np.arange(-2, 10, 2)  # Eb/N0范围：-2,0,2,4,6,8 dB
Ebn0 = 10 ** (EbN0_indB / 10.0)
E = 1
times = 50  # 50次实验
err_rate = np.zeros((len(EbN0_indB), len(N), times))

# ========== 仿真计算（保留原逻辑，仅聚焦目标N） ==========
print("开始仿真计算...")
for i_N in range(len(N)):
    for i_snr in range(len(EbN0_indB)):
        sigma = E / np.sqrt(2 * Ebn0[i_snr])
        for i_t in range(times):
            # 生成比特+BPSK调制+加噪声+判决
            source = np.random.rand(N[i_N]) > 0.5  # 0/1比特
            x = 1 - 2 * source  # BPSK：0→1，1→-1
            noise = np.random.randn(N[i_N]) * sigma
            y = E * x + noise
            result = y <= 0  # 判决：y≤0→1，y>0→0
            error_num = np.sum(result != source)
            err_rate[i_snr, i_N, i_t] = error_num / N[i_N]

# ========== 提取目标数据：N=10000的50次实验误码率 ==========
ber_50times = err_rate[:, target_N_idx, :]  # 形状：(6个Eb/N0, 50次实验)
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))  # 理论误码率
ber_mean = np.mean(ber_50times, axis=1)  # 50次实验的均值误码率
ber_std = np.std(ber_50times, axis=1)    # 50次实验的标准差（量化吻合程度）

# ========== 绘图：50条曲线 + 理论曲线 + 均值曲线 ==========
plt.figure(figsize=(10, 6))
# 1. 绘制50次实验的曲线（浅灰色，降低透明度，避免重叠遮挡）
for i in range(times):
    plt.semilogy(EbN0_indB, ber_50times[:, i],
                 color='lightgray', alpha=0.5, linewidth=1)
# 2. 绘制理论曲线（绿色，突出）
plt.semilogy(EbN0_indB, pe_theory, 'g-', marker='^',
             linewidth=2, label='理论误码率')
# 3. 绘制50次实验的均值曲线（红色，突出）
plt.semilogy(EbN0_indB, ber_mean, 'r-', marker='o',
             linewidth=2, label='50次实验均值误码率')

# ========== 图表美化 ==========
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.xlabel('Eb/N0 (dB)', fontsize=12)
plt.ylabel('误码率 (BER)', fontsize=12)
plt.title('50次实验（10000比特/次）的误码率曲线对比', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.ylim(1e-6, 1e0)  # 固定纵坐标范围，更易对比

# ========== 打印统计信息：各Eb/N0下的标准差（量化吻合程度） ==========
print("\n各Eb/N0下50次实验误码率的标准差（越小表示吻合度越高）：")
for i, ebno in enumerate(EbN0_indB):
    print(f"Eb/N0 = {ebno} dB: 标准差 = {ber_std[i]:.6f}, 均值BER = {ber_mean[i]:.6f}, 理论BER = {pe_theory[i]:.6f}")

plt.show()