import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. 参数设置
fs = 1000       # 采样频率
n_samples = 20000
order = 20      # 实验要求的20阶
f0 = 0.01       # 截止频率系数

# 2. 设计滤波器
# 目标幅度响应：在 f < f0 时为 0，在 f >= f0 时为 1/sqrt(f)
freqs = np.linspace(0, 1, 100)
magnitude = np.where(freqs < f0, 0, 1 / np.sqrt(freqs + 1e-6))
magnitude = magnitude / np.max(magnitude) # 归一化增益

# 使用 firwin2 设计一个高阶 FIR 来逼近该形状（Python 中替代 yulewalk 的常用方案）
b_filter = signal.firwin2(101, freqs, magnitude)

# 3. 生成独立高斯白噪声
white_noise = np.random.normal(0, 1, n_samples)

# 4. 通过滤波器产生闪烁噪声
flicker_noise = signal.lfilter(b_filter, [1.0], white_noise)

# --- 统计方法计算 ---

# 5. 计算自相关函数 (ACF)
# 使用 np.correlate 进行统计估算
autocorr = np.correlate(flicker_noise, flicker_noise, mode='full')
autocorr = autocorr[len(autocorr)//2:] / len(flicker_noise) # 取正半部分并归一化

# 6. 计算功率谱密度 (PSD)
# 使用 Welch 法进行统计功率谱估计
f_psd, p_psd = signal.welch(flicker_noise, fs=1.0, nperseg=1024)

# 7. 绘图可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))

# 时域波形
plt.subplot(3, 1, 1)
plt.plot(flicker_noise[:1000])
plt.title("闪烁噪声 ($1/f$) 时域序列")

# 自相关函数图
plt.subplot(3, 1, 2)
plt.plot(autocorr[:200])
plt.title("统计自相关函数 $R_{xx}(\\tau)$")
plt.xlabel("延迟 (Lag)")

# 功率谱密度图 (双对数坐标)
plt.subplot(3, 1, 3)
plt.loglog(f_psd[1:], p_psd[1:], label='统计 PSD')
plt.loglog(f_psd[1:], 0.01/f_psd[1:], '--', label='理论 $1/f$ 斜率')
plt.title("统计功率谱密度 (双对数坐标)")
plt.xlabel("频率")
plt.ylabel("功率/Hz")
plt.legend()

plt.tight_layout()
plt.savefig('统计方法_有色高斯噪声.png')
plt.show()