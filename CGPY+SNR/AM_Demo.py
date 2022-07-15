import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import scipy.io
from AM_Module import AM_1D
from SNR_Module import snr_fun

# Load our signal
data = scipy.io.loadmat('./testSig3.mat')

x = data['testSig3']
x = np.transpose(x)
N = len(x)

# Add some noise to it
sigma = 10
A = np.eye(N)
b = np.dot(A, x) + sigma * np.random.randn(len(x), 1)

# Set Iterations for AM algorithm
Nit = 500

# Parameters for AM TV Denoising
mu = 0.065
rho = 1.5
tol = 1e-08

# Run the TV-solver
out_FR = AM_1D(b, A, mu, rho, Nit, tol, method="FR")
out_PPR = AM_1D(b, A, mu, rho, Nit, tol, method="PPR")
out_HS = AM_1D(b, A, mu, rho, Nit, tol, method="HS")
out_DY = AM_1D(b, A, mu, rho, Nit, tol, method="DY")

SNR_FR = snr_fun(out_FR["sol"], b)
SNR_PPR = snr_fun(out_PPR["sol"], b)
SNR_HS = snr_fun(out_HS["sol"], b)
SNR_DY = snr_fun(out_DY["sol"], b)

print("SNR_FR: ", SNR_FR)
print("SNR_PPR: ", SNR_PPR)
print("SNR_HS: ", SNR_HS)
print("SNR_DY: ", SNR_DY)


# Calculate Root-Mean Square Error
# rmse_AM = np.sqrt(((x - out["sol"]) ** 2).mean(axis=0))

# Some plotting options to show our results

fopt_FR = np.amin(out_FR["funVal"])
fopt_PPR = np.amin(out_PPR["funVal"])
fopt_HS = np.amin(out_HS["funVal"])
fopt_DY = np.amin(out_DY["funVal"])

# Create Figure 0
fig, ax = plt.subplots(6, 1)

# Plot Original Signal
ax[0].plot(x, linewidth=2.5)
ax[0].set_title('Original signal', fontsize=15)

# Plot Noisy Signal
ax[1].plot(b, linewidth=2.5)
ax[1].set_title('Noisy signal', fontsize=15)

# Plot FR Output
ax[2].plot(out_FR["sol"], linewidth=2.5)
ax[2].set_title('AM TV Denoised (FR)', fontsize=15)

# Plot PPR Output
ax[3].plot(out_PPR["sol"], linewidth=2.5)
ax[3].set_title('AM TV Denoised (PPR)', fontsize=15)

# Plot HS Output
ax[4].plot(out_HS["sol"], linewidth=2.5)
ax[4].set_title('AM TV Denoised (HS)', fontsize=15)

# Plot DY Output
ax[5].plot(out_DY["sol"], linewidth=2.5)
ax[5].set_title('AM TV Denoised (DY)', fontsize=15)

# Create Figure 1
fig2, ax2 = plt.subplots(2, 4)

ax2[0, 0].semilogy(out_FR["funVal"] - fopt_FR, linewidth=3.5, color='blue')
ax2[0, 0].set_xlabel('Iterations', fontsize=15)
ax2[0, 0].set_ylabel('$F(x) - F*$', fontsize=15)
ax2[0, 0].legend('Alternating Minimization TV')
ax2[0, 0].set_title('FR', fontsize=15)
ax2[0, 0].grid('on')

ax2[1, 0].semilogy(out_FR["relativeError"], linewidth=3.5, color='blue')
ax2[1, 0].set_xlabel('Iterations', fontsize=15)
ax2[1, 0].set_ylabel('Relative Err', fontsize=15)
ax2[1, 0].legend('Alternating Minimization TV')
ax2[1, 0].grid('on')

ax2[0, 1].semilogy(out_PPR["funVal"] - fopt_PPR, linewidth=3.5, color='blue')
ax2[0, 1].set_xlabel('Iterations', fontsize=15)
ax2[0, 1].set_ylabel('$F(x) - F*$', fontsize=15)
ax2[0, 1].legend('Alternating Minimization TV')
ax2[0, 1].set_title('PPR', fontsize=15)
ax2[0, 1].grid('on')

ax2[1, 1].semilogy(out_PPR["relativeError"], linewidth=3.5, color='blue')
ax2[1, 1].set_xlabel('Iterations', fontsize=15)
ax2[1, 1].set_ylabel('Relative Err', fontsize=15)
ax2[1, 1].legend('Alternating Minimization TV')
ax2[1, 1].grid('on')

ax2[0, 2].semilogy(out_HS["funVal"] - fopt_HS, linewidth=3.5, color='blue')
ax2[0, 2].set_xlabel('Iterations', fontsize=15)
ax2[0, 2].set_ylabel('$F(x) - F*$', fontsize=15)
ax2[0, 2].legend('Alternating Minimization TV')
ax2[0, 2].set_title('HS', fontsize=15)
ax2[0, 2].grid('on')

ax2[1, 2].semilogy(out_HS["relativeError"], linewidth=3.5, color='blue')
ax2[1, 2].set_xlabel('Iterations', fontsize=15)
ax2[1, 2].set_ylabel('Relative Err', fontsize=15)
ax2[1, 2].legend('Alternating Minimization TV')
ax2[1, 2].grid('on')

ax2[0, 3].semilogy(out_DY["funVal"] - fopt_DY, linewidth=3.5, color='blue')
ax2[0, 3].set_xlabel('Iterations', fontsize=15)
ax2[0, 3].set_ylabel('$F(x) - F*$', fontsize=15)
ax2[0, 3].legend('Alternating Minimization TV')
ax2[0, 3].set_title('DY', fontsize=15)
ax2[0, 3].grid('on')

ax2[1, 3].semilogy(out_DY["relativeError"], linewidth=3.5, color='blue')
ax2[1, 3].set_xlabel('Iterations', fontsize=15)
ax2[1, 3].set_ylabel('Relative Err', fontsize=15)
ax2[1, 3].legend('Alternating Minimization TV')
ax2[1, 3].grid('on')

fig3, ax = plt.subplots()

algos = ['FR', 'PPR', 'HS', 'DY']
SNR_val = [round(SNR_FR, 5), round(SNR_PPR, 5), round(SNR_HS, 5), round(SNR_DY, 5)]

low = min(SNR_val)
high = max(SNR_val)
plt.ylim(low, high)

rects = ax.bar(algos, SNR_val)

for rect, label in zip(rects, SNR_val):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

fig.tight_layout()
fig2.tight_layout()

plt.show()
