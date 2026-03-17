import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def f(x):
    return 20 * x * (1 - x)**3

M = 2.109375
num_needed = 100000

accepted_a = []
total_tried_a = 0

while len(accepted_a) < num_needed:
    total_tried_a += 1
    x_cand = np.random.uniform(0, 1)
    u = np.random.uniform(0, M)
    if u <= f(x_cand):
        accepted_a.append(x_cand)

m1, m2 = f(0.25), f(0.5)
area1 = m1 * 0.5
area2 = m2 * 0.5
total_area = area1 + area2
p1 = area1 / total_area

accepted_b = []
total_tried_b = 0

while len(accepted_b) < num_needed:
    total_tried_b += 1
    if np.random.uniform(0, 1) < p1:
        x_cand = np.random.uniform(0, 0.5)
        limit = m1
    else:
        x_cand = np.random.uniform(0.5, 1)
        limit = m2
    u = np.random.uniform(0, limit)
    if u <= f(x_cand):
        accepted_b.append(x_cand)

efficiency_a = num_needed / total_tried_a
efficiency_b = num_needed / total_tried_b

fig = plt.figure(figsize=(14, 10))

x = np.linspace(0, 1, 500)
y = f(x)

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = 20x(1-x)^3')
ax1.axhline(y=M, color='r', linestyle='--', linewidth=1.5, label='Envelope M = %.4f' % M)
ax1.fill_between(x, 0, y, alpha=0.3, color='green', label='Accept region')
ax1.fill_between(x, y, M, alpha=0.3, color='red', label='Reject region')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, M + 0.2)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Rejection Region (Rectangular Envelope)', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y, 'b-', linewidth=2, label='Target f(x)')
ax2.axhline(y=m1, color='orange', linestyle='--', linewidth=1.5, label='m1 = %.4f' % m1)
ax2.axhline(y=m2, color='purple', linestyle='--', linewidth=1.5, label='m2 = %.4f' % m2)
ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
mask1 = x <= 0.5
mask2 = x > 0.5
ax2.fill_between(x[mask1], 0, y[mask1], alpha=0.3, color='green')
ax2.fill_between(x[mask1], y[mask1], m1, alpha=0.3, color='red')
ax2.fill_between(x[mask2], 0, y[mask2], alpha=0.3, color='green')
ax2.fill_between(x[mask2], y[mask2], m2, alpha=0.3, color='red')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, M + 0.2)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Rejection Region (Piecewise Envelope)', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(accepted_a, bins=50, density=True, alpha=0.7, 
         color='steelblue', edgecolor='black', label='Sampled data')
ax3.plot(x, y, 'r-', linewidth=2, label='Theoretical f(x)')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, M + 0.2)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Distribution (Rectangular Rejection)\nEfficiency: %.2f%%' % (efficiency_a*100), fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 2, 4)
ax4.hist(accepted_b, bins=50, density=True, alpha=0.7, 
         color='coral', edgecolor='black', label='Sampled data')
ax4.plot(x, y, 'r-', linewidth=2, label='Theoretical f(x)')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, M + 0.2)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('Distribution (Piecewise Rejection)\nEfficiency: %.2f%%' % (efficiency_b*100), fontsize=14)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('舍选法可视化.png', dpi=150, bbox_inches='tight')
plt.show()

print("Rectangular rejection efficiency: %.2f%%" % (efficiency_a*100))
print("Piecewise rejection efficiency: %.2f%%" % (efficiency_b*100))
print("Image saved as: rejection_sampling.png")
