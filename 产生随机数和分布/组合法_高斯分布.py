import numpy as np
import matplotlib.pyplot as plt

# 1. Set Parameters from the table
p = [1 / 2, 1 / 3, 1 / 6]  # Weights
a = [-1, 0, 1]  # Means (ai)
b = [1 / 4, 1, 1 / 2]  # Standard Deviations (bi)
n_samples = 10000


def box_muller():
    """Generates a standard normal sample N(0,1)"""
    u1, u2 = np.random.rand(), np.random.rand()
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0


# 2. Generate samples using the Composition Method
samples = []
for _ in range(n_samples):
    # Step A: Choose which Gaussian component to use based on p_i
    rand_p = np.random.rand()
    if rand_p < p[0]:
        i = 0
    elif rand_p < p[0] + p[1]:
        i = 1
    else:
        i = 2

    # Step B: Generate sample from chosen Gaussian N(ai, bi^2)
    z = box_muller()
    x = a[i] + b[i] * z
    samples.append(x)

# 3. Visualization
plt.figure(figsize=(10, 6))

# Plot the simulated histogram
count, bins, ignored = plt.hist(samples, bins=100, density=True,
                                alpha=0.6, color='skyblue', label='Simulated Samples')

# Plot the theoretical PDF for comparison
x_axis = np.linspace(-4, 4, 1000)
theoretical_pdf = np.zeros_like(x_axis)
for i in range(3):
    # Gaussian PDF formula: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x-mu)/sigma)^2)
    component = (1 / (b[i] * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((x_axis - a[i]) / b[i]) ** 2)
    theoretical_pdf += p[i] * component

plt.plot(x_axis, theoretical_pdf, 'r-', lw=2, label='Theoretical PDF')
plt.title('Gaussian Mixture Model (Composition Method)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('Gaussian Mixture Model (Composition Method).png')
plt.show()