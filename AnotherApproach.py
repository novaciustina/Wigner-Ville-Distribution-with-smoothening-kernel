import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg, ndimage

# Data reading
data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64)
# Define the sampling frequency and time vector
duration = 660         # seconds
n_samples = 16500000   # total number of samples
sampling_rate = 25000  # sampling frequency in Hz - ALREADY DOWNSAMPLED
chunk_size = 2048 # used to manage memory usage, but in practice, you would want to use a larger chunk size for better frequency resolution
stride = 1024  # used to overlap chunks

#Instead of using the Hankel matrix, I iwll just try a loop over time
#This just calculates the autocorelation function 
def wigner_ville_distribution(x, max_tau):
    """Compute the Wigner-Ville distribution of a signal x with a maximum lag of max_tau. The Wigner-Ville distribution is a time-frequency representation that provides high resolution in both time and frequency domains. The max_tau parameter controls the maximum lag for the autocorrelation function, which affects the frequency resolution of the distribution. A larger max_tau will provide better frequency resolution but may introduce more cross-terms and require more computational resources."""
    N = len(x) #Number of samples in the input signal

    wvd = np.zeros((N, 2*max_tau+1), dtype=np.complex64)

    for t in range(max_tau, N - max_tau):
        # We go from -max_tau to max_tau to get the lags for the autocorrelation function, and we need to make sure we don't go out of bounds of the signal
        # This is done for my computer to not explode, but in practice, you would want to use a larger max_tau for better frequency resolution, and you would need to handle the edge effects properly (e.g., by zero-padding the signal)
        tau = np.arange(-max_tau, max_tau + 1)
        # Create the autocorrelation function for the current time point t and lag tau
        r = x[t + tau] * np.conj(x[t - tau])
        # Compute the Fourier transform of the autocorrelation function to get the Wigner-Ville distribution at time t
        wvd[t, :] = np.fft.fftshift(np.fft.fft(r))

    return np.real(wvd) #Return the real here, not because the WVD is notreal, but to deal with numerical errors that may introduce small imaginary components

# Compute the Wigner-Ville distribution for the entire signal in chunks to manage memory usage
wvd_chunks = []

for start in range(0, len(data) - chunk_size, stride):
    chunk = data[start:start + chunk_size]
    wvd_chunk = wigner_ville_distribution(chunk, max_tau=256)
    wvd_chunks.append(wvd_chunk)

# Concatenate the WVD chunks to get the full Wigner-Ville distribution for the entire signal
wvd = np.concatenate(wvd_chunks, axis=0)

# Plot the Wigner-Ville distribution
num_time = wvd.shape[0]
num_freq = wvd.shape[1]

# Create the time and frequency axes for plotting
time_axis = np.arange(num_time) * stride / sampling_rate
freqs = np.fft.fftshift(np.fft.fftfreq(num_freq, d=1 / sampling_rate))

# Magnitude + log scale
wvd_db = 10 * np.log10(np.abs(wvd) + 1e-20)

# Dynamic range clipping (better contrast)
vmin = np.percentile(wvd_db, 5)
vmax = np.percentile(wvd_db, 99)

plt.figure(figsize=(14, 7))

plt.imshow(wvd_db.T, aspect='auto', origin='lower', extent=[ time_axis[0], time_axis[-1], freqs[0], freqs[-1]  ], vmin=vmin, vmax=vmax)
plt.colorbar(label="Power (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Wigner–Ville Distribution (Chunked, Smoothed)")

plt.tight_layout()
plt.show()