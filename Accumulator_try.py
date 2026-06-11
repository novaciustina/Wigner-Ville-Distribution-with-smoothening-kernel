import numpy as np
import matplotlib.pyplot as plt
import h5py  
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from StraightenedSignal import straighten_signal
from WaterfallCurveData import update_waterfall_curve
import psutil
#from MovingAverage import moving_average
# Data reading
data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64) # Data from L0 Products, which has the IQ data, already downsampled to 25 kHz

# Sampling parameters - Diffrent for each satellite
sampling_rate = 25000  # HZ, ALREADY DOWNSAMPLED
dt = 1 / sampling_rate # s, sampling time
duration = 660 #s
baud_rate = 1200 

# Processing parameters - Diffrent for each satellite
chunk_size  = 1200         # controls how many samples we process at once (bigger = better frequency resolution, but more memory)
max_tau = 256 * 2           # maximum lag in samples for WVD calculation. controls the frequency resolution.
stride = chunk_size - 2*max_tau   # overlap between chunks to ensure we capture all time points without edge effects - TO NOT LOSE ENERGY


# Actual WVD calculation function
def wigner_ville_distribution(x, max_tau):
    # instead of looping over each time point t, we build matrices of indices to compute all t+tau and t-tau combinations at once, then do a single FFT across the lag dimension for all time points simultaneously
    N  = len(x)
    taus = np.arange(-max_tau, max_tau + 1)       # lag vector, shape (2*max_tau+1,)
    t_idx = np.arange(max_tau, N - max_tau)       # time indeces, shape (n_t,)
    plus_idx  = t_idx[:, None] + taus[None, :]    # all t+tau combinations,  shape:  (n_t, 2*max_tau+1)
    minus_idx = t_idx[:, None] - taus[None, :]    # all t-tau combinations, shape:  (n_t, 2*max_tau+1)
    R = x[plus_idx] * np.conj(x[minus_idx])       # compute the full autocorrelation matrix, shape: (n_t, 2*max_tau+1)
    # Apply Hanning window across lag axis (axis=1) to suppress cross-terms
    window = np.hanning(2 * max_tau + 1).astype(np.float32)
    R = R * window[None, :]                        # broadcast correctly over all time points
    wvd = np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)  
    return np.real(wvd).astype(np.float32)  # cast to float32 here to halve memory

# Compute total chunks for progress display 
total_chunks = (len(data) - chunk_size) // stride + 1
t_chunk = chunk_size / sampling_rate  # duration of chunk in seconds
print(f"Total chunks to process: {total_chunks}")
print(f"Each chunk covers {t_chunk:.2f}s of signal")
print("Starting...")


cnr_per_chunk = []  # one value per chunk pair
prev_wvd = None

for i in range(total_chunks):
    start = i * stride
    chunk = data[start : start + chunk_size]
    wvd_chunk = wigner_ville_distribution(chunk, max_tau=max_tau)
    wvd_chunk = gaussian_filter1d(wvd_chunk, sigma=3, axis=0)

    if prev_wvd is not None:
        # Eq 5: total power — use abs to handle negative WVD values
        # Px = 0.5 * (np.abs(prev_wvd)**2 + np.abs(wvd_chunk)**2).astype(np.float64)
        Px = wvd_chunk**2  # power of current chunk, shape: (n_time, n_freq)
        # Eq 6: signal power — cross product
        #Ps = (prev_wvd * wvd_chunk).astype(np.float64)
        Ps = 0.5 * ( prev_wvd * np.conj(wvd_chunk) + wvd_chunk * np.conj(prev_wvd) ).astype(np.float64)  
        # Eq 7: noise power
        Pn = np.maximum(Px - Ps, 1e-20) # use max to avoid negative or zero noise power
        # Eq 8: SNR map for this chunk pair, shape (n_time, n_freq)
        cnb_chunk = Ps / Pn

        # take mean over the whole map — not max, which picks outliers
        cnr_per_chunk.append(np.mean(cnb_chunk))


    prev_wvd = wvd_chunk.copy()
    if i % 50 == 0:  # print every 50 chunks, not every chunk
            print(f"  chunk {i}/{total_chunks}  ({100*i/total_chunks:.1f}%) ")
            print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

# Convert to dB
CNB_1d = 10 * np.log10(np.array(cnr_per_chunk) + 1e-20)
np.savetxt("cnr_per_chunk_autocorrelation.csv", cnr_per_chunk, delimiter=",")
time_CNB = np.linspace(0, duration, len(CNB_1d))

plt.figure(figsize=(12, 4))
plt.plot(time_CNB, CNB_1d)
plt.xlabel("Time (s)")
plt.ylabel("CNB (dB)")
plt.title("CNB vs Time") #(per chunk pair, peak over time-frequency)
plt.grid(True)
plt.tight_layout()
plt.show()