import numpy as np
import matplotlib.pyplot as plt


def run_simulation(total_n, interval=1000):
    # 初始化变量
    points_inside = 0
    results_n = []
    results_pi = []
    results_abs_error = []
    results_theoretical_error = []

    print(f"{'次数 (N)':<12} | {'π 估值':<10} | {'实际绝对误差':<12} | {'95%理论误差界'}")
    print("-" * 60)

    for i in range(1, total_n + 1):
        # 生成随机点并判断
        x, y = np.random.rand(), np.random.rand()
        if x ** 2 + y ** 2 <= 1:
            points_inside += 1

        # 每隔 interval 次输出并记录一次数据
        if i % interval == 0 or i == 1:
            pi_hat = 4 * (points_inside / i)
            abs_error = abs(pi_hat - np.pi)

            # 中心极限定理估算的绝对误差 (95% 置信度, z ≈ 1.96)
            p_hat = points_inside / i
            # 误差 = 4 * 1.96 * sqrt(p*(1-p)/n)
            theoretical_error = 4 * 1.96 * np.sqrt((p_hat * (1 - p_hat)) / i)

            results_n.append(i)
            results_pi.append(pi_hat)
            results_abs_error.append(abs_error)
            results_theoretical_error.append(theoretical_error)

            print(f"{i:<12} | {pi_hat:.6f} | {abs_error:.6f} | {theoretical_error:.6f}")

    return results_n, results_abs_error, results_theoretical_error


# 执行仿真：总次数 100,000，每 2000 次输出一次
n_vals, abs_errors, theoretical_errors = run_simulation(100000, 2000)

# 绘制误差关系曲线
plt.figure(figsize=(10, 5))
plt.plot(n_vals, abs_errors, label='Actual Absolute Error', color='blue', alpha=0.6)
plt.plot(n_vals, theoretical_errors, label='Theoretical Error (95% CLT)', color='red', linestyle='--')
plt.xlabel('Number of Iterations (N)')
plt.ylabel('Absolute Error')
plt.title('Relationship between Iterations and Absolute Error')
plt.legend()
plt.grid(True)
plt.show()