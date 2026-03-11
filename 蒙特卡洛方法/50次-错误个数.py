import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ========== 基础配置：解决负号显示 + 字体 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 仿真参数 ==========
EbN0_indB = np.arange(-2, 10, 2)  # Eb/N0范围：-2,0,2,4,6,8 dB
Ebn0 = 10 ** (EbN0_indB / 10.0)
E = 1
times = 50  # 50次实验
error_threshold = 100  # 错误个数≥100时停止单次实验

# 存储结果：误码率、传输比特数（维度：[Eb/N0数, 实验次数]）
ber_dynamic = np.zeros((len(EbN0_indB), times))
bits_transmitted = np.zeros((len(EbN0_indB), times))

# ========== 动态比特数仿真（错误数≥100停止） ==========
print("开始动态比特数仿真...")
for i_snr in range(len(EbN0_indB)):
    sigma = E / np.sqrt(2 * Ebn0[i_snr])  # 噪声标准差
    print(f"正在仿真 Eb/N0 = {EbN0_indB[i_snr]} dB ...")
    for i_t in range(times):
        error_count = 0  # 累计错误数
        total_bits = 0  # 累计传输比特数
        while error_count < error_threshold:
            # 每次生成1000比特（批量生成，提升效率）
            batch_bits = 1000
            total_bits += batch_bits

            # BPSK调制 + 加噪声 + 判决
            source = np.random.rand(batch_bits) > 0.5  # 0/1比特
            x = 1 - 2 * source  # BPSK调制：0→1，1→-1
            noise = np.random.randn(batch_bits) * sigma
            y = E * x + noise
            result = y <= 0  # 判决：y≤0→1，y>0→0

            # 累计错误数
            error_count += np.sum(result != source)

        # 计算本次实验的误码率，存储结果
        ber_dynamic[i_snr, i_t] = error_count / total_bits
        bits_transmitted[i_snr, i_t] = total_bits

# ========== 理论误码率 + 统计指标计算 ==========
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))  # 理论误码率
ber_mean = np.mean(ber_dynamic, axis=1)  # 50次实验误码率均值
ber_std = np.std(ber_dynamic, axis=1)  # 50次实验误码率标准差
bits_mean = np.mean(bits_transmitted, axis=1)  # 各Eb/N0平均传输比特数
bits_std = np.std(bits_transmitted, axis=1)  # 各Eb/N0传输比特数标准差

# ========== 绘图：50条动态比特数实验的误码率曲线 ==========
plt.figure(figsize=(10, 6))
# 1. 绘制50次实验的曲线（浅灰色，降低透明度）
for i in range(times):
    plt.semilogy(EbN0_indB, ber_dynamic[:, i],
                 color='lightgray', alpha=0.5, linewidth=1)
# 2. 绘制理论曲线（绿色）
plt.semilogy(EbN0_indB, pe_theory, 'g-', marker='^',
             linewidth=2, label='理论误码率')
# 3. 绘制50次实验均值曲线（红色）
plt.semilogy(EbN0_indB, ber_mean, 'r-', marker='o',
             linewidth=2, label='50次实验均值误码率')

# 图表美化
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.xlabel('Eb/N0 (dB)', fontsize=12)
plt.ylabel('误码率 (BER)', fontsize=12)
plt.title('50次实验（错误数≥100停止）的误码率曲线对比', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.ylim(1e-6, 1e0)

# ========== 打印统计结果 ==========
print("\n===== 各Eb/N0下50次实验统计结果 =====")
print(f"停止条件：错误个数 ≥ {error_threshold} 比特")
print("-" * 80)
print(f"{'Eb/N0(dB)':<10} {'平均传输比特数':<15} {'比特数标准差':<15} {'误码率均值':<15} {'误码率标准差':<15}")
print("-" * 80)
for i, ebno in enumerate(EbN0_indB):
    print(f"{ebno:<10} {bits_mean[i]:<15.0f} {bits_std[i]:<15.0f} {ber_mean[i]:<15.6f} {ber_std[i]:<15.6f}")

# ========== 绘制传输比特数统计柱状图 ==========
plt.figure(figsize=(10, 5))
x = np.arange(len(EbN0_indB))
width = 0.35
plt.bar(x - width / 2, bits_mean, width, label='平均传输比特数', yerr=bits_std, capsize=5)
plt.xticks(x, EbN0_indB)
plt.xlabel('Eb/N0 (dB)', fontsize=12)
plt.ylabel('传输比特数', fontsize=12)
plt.title('各Eb/N0下50次实验的平均传输比特数（错误数≥100停止）', fontsize=14)
plt.grid(axis='y', linestyle='-', linewidth=0.5)
plt.legend()

plt.show()