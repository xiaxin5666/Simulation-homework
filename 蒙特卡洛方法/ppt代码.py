import numpy as np
import matplotlib.pyplot as plt
m = int(input('请输入实验次数 M: '))
n = int(input('请输入每次实验的投点数 N: '))
z = np.zeros(m)
data = np.zeros((n, m))
for j in range(m):
    x = np.random.rand(n)
    y = np.random.rand(n)
    k = 0
    for i in range(n):
        if x[i] ** 2 + y[i] ** 2 <= 1:
            k += 1
        data[i, j] = 4 * (k / (i + 1))
    z[j] = data[n - 1, j]
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for j in range(m):
    plt.plot(range(1, n + 1), data[:, j], 'r',
    alpha=0.1)
plt.xlabel('Point number')
plt.ylabel('π')
plt.title(f'M={m}, N={n}')
plt.grid(True, alpha=0.3)
plt.axhline(y=np.pi, color='k', linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(range(1, m + 1), z, 'b-')
plt.xlabel('Trial number')
plt.ylabel('π')
plt.title(f'Finalestimation(Mean={np.mean(z):.6f})')
plt.grid(True, alpha=0.3)
plt.axhline(y=np.pi, color='k', linestyle='--')
plt.axhline(y=np.mean(z), color='r',
linestyle=':')
plt.tight_layout()
plt.show()
print(f"\nπ的真实值: {np.pi}")
print(f"π的估计值: {np.mean(z)}")
print(f"误差: {abs(np.mean(z) - np.pi)}")
print(f"z的标准差: {np.std(z, ddof=1)}")
