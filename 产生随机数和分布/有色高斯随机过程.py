import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. Setup Parameters
fs = 1000
n_samples = 20000
order = 20
f0 = 0.01  # Cutoff frequency ratio

# 2. Design the Filter using Frequency Sampling (firwin2)
# Since scipy lacks yulewalk, we create a target response and
# use 'butter' or 'cheby' for specific shapes, or 'firwin2' for arbitrary.
# For a 20th order IIR, we can approximate the 1/f slope like this:
num_taps = 101
freqs = np.linspace(0, 1, 100)
# Target: 0 below f0, 1/sqrt(f) above f0
magnitude = np.where(freqs < f0, 0, 1 / np.sqrt(freqs + 1e-6))
magnitude = magnitude / np.max(magnitude) # Normalize gain

# Design an FIR filter (easier to control shape)
b_fir = signal.firwin2(num_taps, freqs, magnitude)

# If you STRICTLY need an IIR 20th order, we use Prony's method or
# similar via external tools, but for this experiment,
# a high-order FIR or a Butterworth/Cheby combination is more common in Python.
# Let's use a Butterworth Highpass to handle the f < f0 requirement:
b_hp, a_hp = signal.butter(4, f0, btype='highpass')

# 3. Generate Noise
white_noise = np.random.normal(0, 1, n_samples)

# 4. Apply Filters
# First, apply the highpass (f < f0 -> 0)
step1 = signal.lfilter(b_hp, a_hp, white_noise)
# Then apply the 1/sqrt(f) shape
flicker_noise = signal.lfilter(b_fir, [1.0], step1)

# 5. Analysis (PSD)
f_psd, p_psd = signal.welch(flicker_noise, fs, nperseg=1024)

# 6. Plotting
plt.figure(figsize=(10, 5))
plt.loglog(f_psd[1:], p_psd[1:], label='Generated Noise')
plt.plot(f_psd[1:], 0.01/f_psd[1:], '--', label='Theoretical 1/f Slope', alpha=0.7)
plt.title("Power Spectral Density of Generated Flicker Noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('Power Spectral Density of Generated Flicker Noise.png')
plt.show()