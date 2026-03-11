import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

  # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 基础参数配置
EbN0_indB = np.arange(-2, 10, 2)  # Eb/N0(dB) 区间
Ebn0 = 10 ** (EbN0_indB / 10.0)
E = 1  # 信号能量
times = 50  # 每个参数下的独立仿真次数
target_error = 100  # 固定错误数法的目标错误数
max_symbols = 10 ** 7  # 固定错误数法的最大符号数（防止无限循环）
N_fixed_sym = np.array([1000, 10000, 100000, 1000000])  # 固定符号数法的符号数


# ------------------------------
# 方法1：固定符号数法（原逻辑封装）
# ------------------------------
def fixed_symbols_simulation(N_list, EbN0_indB, E, times):
    Ebn0 = 10 ** (EbN0_indB / 10.0)
    err_rate = np.zeros((len(Ebn0), len(N_list), times))

    for i_N in range(len(N_list)):
        N = N_list[i_N]
        for i_snr in range(len(Ebn0)):
            sigma = E / np.sqrt(2 * Ebn0[i_snr])
            for i_t in range(times):
                # 生成固定数量的符号
                r = np.random.rand(N)
                source = r > 0.5
                x = 1 - 2 * source  # BPSK调制
                noise = np.random.randn(N) * sigma
                y = E * x + noise
                result = y <= 0  # 判决
                error_num = np.sum(result != source)
                err_rate[i_snr, i_N, i_t] = error_num / N
    return err_rate


# ------------------------------
# 方法2：固定错误数法
# ------------------------------
def fixed_errors_simulation(target_err, max_sym, EbN0_indB, E, times):
    Ebn0 = 10 ** (EbN0_indB / 10.0)
    err_rate = np.zeros((len(Ebn0), times))
    total_syms = np.zeros((len(Ebn0), times))  # 记录实际传输的符号数

    for i_snr in range(len(Ebn0)):
        sigma = E / np.sqrt(2 * Ebn0[i_snr])
        for i_t in range(times):
            error_count = 0
            symbol_count = 0
            # 循环直到错误数达标或达到最大符号数
            while error_count < target_err and symbol_count < max_sym:
                # 每次生成一批符号（批量处理提升效率）
                batch_size = 1000
                r = np.random.rand(batch_size)
                source = r > 0.5
                x = 1 - 2 * source
                noise = np.random.randn(batch_size) * sigma
                y = E * x + noise
                result = y <= 0
                batch_errors = np.sum(result != source)

                error_count += batch_errors
                symbol_count += batch_size

            # 计算误码率（若未达到目标错误数，用实际错误数计算）
            if symbol_count == 0:
                err_rate[i_snr, i_t] = 0
            else:
                err_rate[i_snr, i_t] = error_count / symbol_count
            total_syms[i_snr, i_t] = symbol_count
    return err_rate, total_syms


# ------------------------------
# 执行仿真
# ------------------------------
print("开始固定符号数法仿真...")
err_rate_fixed_sym = fixed_symbols_simulation(N_fixed_sym, EbN0_indB, E, times)

print("开始固定错误数法仿真...")
err_rate_fixed_err, total_syms_fixed_err = fixed_errors_simulation(target_error, max_symbols, EbN0_indB, E, times)

# ------------------------------
# 理论误码率计算
# ------------------------------
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))

# ------------------------------
# 结果可视化
# ------------------------------
# 图1：固定符号数法（不同符号数）vs 理论值
plt.figure(figsize=(10, 6))
markers = ['o', '^', 's', 'x']
labels = [f'{n}bits' for n in N_fixed_sym]
for i in range(len(N_fixed_sym)):
    # 取50次仿真的平均误码率
    ber_mean = np.mean(err_rate_fixed_sym[:, i, :], axis=1)
    plt.semilogy(EbN0_indB, ber_mean, color='k', marker=markers[i],
                 markersize=8, linewidth=2, label=labels[i])
plt.semilogy(EbN0_indB, pe_theory, 'r--', linewidth=2, label='Theory')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.xlabel('Eb/N0(dB)', fontsize=14)
plt.ylabel('BER', fontsize=14)
plt.title('固定符号数法 BER 性能', fontsize=16)
plt.legend(fontsize=10)

# 图2：固定错误数法 vs 理论值
plt.figure(figsize=(10, 6))
ber_fixed_err_mean = np.mean(err_rate_fixed_err, axis=1)
ber_fixed_err_std = np.std(err_rate_fixed_err, axis=1)
# 绘制均值+误差棒
plt.errorbar(EbN0_indB, ber_fixed_err_mean, yerr=ber_fixed_err_std,
             fmt='ko-', linewidth=2, markersize=8, label=f'Fixed Errors (target={target_error})')
plt.semilogy(EbN0_indB, pe_theory, 'r--', linewidth=2, label='Theory')
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.xlabel('Eb/N0(dB)', fontsize=14)
plt.ylabel('BER', fontsize=14)
plt.title('固定错误数法 BER 性能', fontsize=16)
plt.legend(fontsize=10)

# 图3：固定错误数法的实际传输符号数
plt.figure(figsize=(10, 6))
syms_mean = np.mean(total_syms_fixed_err, axis=1)
plt.plot(EbN0_indB, syms_mean, 'bo-', linewidth=2, markersize=8)
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.xlabel('Eb/N0(dB)', fontsize=14)
plt.ylabel('Average Symbols Transmitted', fontsize=14)
plt.title('固定错误数法的平均传输符号数', fontsize=16)

plt.show()

# 输出关键统计信息
print("\n===== 固定错误数法统计 =====")
for i, ebno in enumerate(EbN0_indB):
    avg_sym = np.mean(total_syms_fixed_err[i])
    avg_ber = np.mean(err_rate_fixed_err[i])
    print(f"Eb/N0={ebno}dB: 平均传输符号数={avg_sym:.0f}, 平均误码率={avg_ber:.6f}")

print("\n===== 固定符号数法（1e6 bits）统计 =====")
for i, ebno in enumerate(EbN0_indB):
    avg_ber = np.mean(err_rate_fixed_sym[i, -1, :])  # 取1e6 bits的均值
    print(f"Eb/N0={ebno}dB: 平均误码率={avg_ber:.6f}")