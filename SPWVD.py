import numpy as np
import matplotlib.pyplot as plt
import h5py  
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# Data reading
data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64)

# Sampling parameters
sampling_rate = 25000  # ALREADY DOWNSAMPLED
duration = 660 #s

# Processing parameters
chunk_size  = 1024 * 16   # controls how many samples we process at once (bigger = better frequency resolution, but more memory)
stride_skip = 4           # stride_skip: controls how far we jump between chunks (reduces time resolution)
max_tau = 256             # maximum lag in samples for WVD calculation. controls the frequency resolution.
stride = chunk_size // 2  # overlap between chunks to ensure we capture all time points without edge effects

# Power lists
power_time = []   # instantaneous power over time
time_power = []   # corresponding time points for power plot
total_energy = 0  # total energy 

def wigner_ville_distribution(x, max_tau):
    # instead of looping over each time point t, we build matrices of indices to compute all t+tau and t-tau combinations at once, then do a single FFT across the lag dimension for all time points simultaneously
    N  = len(x)
    taus = np.arange(-max_tau, max_tau + 1)       # lag vector, shape (2*max_tau+1,)
    t_idx = np.arange(max_tau, N - max_tau)       # time indeces, shape (n_t,)
    plus_idx  = t_idx[:, None] + taus[None, :]    # all t+tau combinations,  shape:  (n_t, 2*max_tau+1)
    minus_idx = t_idx[:, None] - taus[None, :]    # all t-tau combinations, shape:  (n_t, 2*max_tau+1)
    R = x[plus_idx] * np.conj(x[minus_idx])       # compute the full autocorrelation matrix, shape: (n_t, 2*max_tau+1)
    """ Hanning window for smoothing the WVD, makes things worse 
    tau_window = max_tau 
    window = np.hanning(2 * tau_window + 1).astype(np.float32) # create a Hanning window for smoothing the WVD, shape: (2*max_tau+1,)
    window /= np.sum(window) # normalize window to avoid processing gain changes 
    R = (x[plus_idx] * np.conj(x[minus_idx])) * window[None, :] # apply the window function across the lag dimension (axis=1) to get the PWVD
    """
    wvd = np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)  
    return np.real(wvd).astype(np.float32)  # cast to float32 here to halve memory

# Compute total chunks for progress display 
total_chunks = len(range(0, len(data) - chunk_size, stride * stride_skip))
print(f"Total chunks to process: {total_chunks}")
print(f"Each chunk covers {chunk_size/sampling_rate:.2f}s of signal")
print("Starting...")

#Disk memory allocation 
with h5py.File("wvd_output.h5", "w") as f:  
    dset = f.create_dataset( "wvd", shape=(0, 2 * max_tau + 1), maxshape=(None, 2 * max_tau + 1), dtype=np.float32, chunks=(256, 2 * max_tau + 1)  )
    row = 0  # keep track of how many rows we've written to disk so far
    # compute WVD in chunks and write each chunk to disk immediately to avoid RAM issues
    for i, start in enumerate(range(0, len(data) - chunk_size, stride * stride_skip)):
        chunk     = data[start : start + chunk_size]
        wvd_chunk = wigner_ville_distribution(chunk, max_tau=max_tau)
        wvd_chunk = gaussian_filter1d(wvd_chunk, sigma=5, axis=0)  # smooth along time axis to reduce noise
        # Sigma controls the amount of smoothing, higher = smoother but more blurring
        # Instantaneous power (sum over frequency axis)
        p_chunk = np.sum(wvd_chunk, axis=1)  
        t_idx = np.arange(max_tau, chunk_size - max_tau) 
        global_t = start + t_idx 
        time_power.append(global_t / sampling_rate)
        power_time.append(p_chunk)
        # Total energy (sum over everything)
        total_energy += np.sum(wvd_chunk)
        new_row = row + len(wvd_chunk)
        dset.resize(new_row, axis=0)       # grow disk dataset by this chunk's rows
        dset[row:new_row] = wvd_chunk      # write to disk — chunk freed after this line
        row = new_row  # update our row counter to the new total on disk
        # print every 50 chunks
        if i % 50 == 0:  # print every 50 chunks, not every chunk
            print(f"  chunk {i}/{total_chunks}  ({100*i/total_chunks:.1f}%)  —  {row} rows on disk")

    print(f"Done. Reading {row} rows back from disk for plotting...")
    wvd = dset[:]  # read back from disk into RAM for plotting (now that all processing is done)

# Combine power lists into single array for plotting
power_time = np.concatenate(power_time)
time_power = np.concatenate(time_power)

# Plotting
num_time = wvd.shape[0]
num_freq = wvd.shape[1]

# Visualisation
max_display_rows = 2000                                    # target pixel height
block_size  = max(1, num_time // max_display_rows)         # rows to average together
trimmed     = num_time - (num_time % block_size)           #  trim to exact multiple
wvd_display = wvd[:trimmed].reshape(-1, block_size, num_freq).mean(axis=1)  # average down
print(f"Display shape: {wvd_display.shape}  (downsampled from {num_time} rows)")

# Time and frequency axes
time_axis = np.linspace(0, duration, wvd_display.shape[0])
freqs     = np.fft.fftshift(np.fft.fftfreq(num_freq, d=1 / sampling_rate))

# convert to dB for better visualization
wvd_db = 10 * np.log10(np.abs(wvd_display) + 1e-20)
vmin   = np.percentile(wvd_db, 5)
vmax   = np.percentile(wvd_db, 99)

# Plot the Wigner-Ville distribution
plt.figure(figsize=(14, 7))
#plt.imshow(wvd_db.T, aspect='auto', origin='lower', extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]], vmin=vmin, vmax=vmax)
plt.imshow(wvd_db, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], time_axis[0], time_axis[-1]],  vmin=vmin, vmax=vmax)
plt.colorbar(label="Power (dB)")
plt.gca().invert_yaxis() # invert y-axis to have time increasing downwards
plt.ylabel("Time (s)")
plt.xlabel("Frequency (Hz)")
plt.title("Wigner–Ville Distribution")
plt.tight_layout()
plt.show()

# Plot power over time
plt.figure(figsize=(12, 4))
plt.plot(time_power, 10*np.log10(np.abs(power_time) + 1e-20))
plt.xlabel("Time (s)")
plt.ylabel("Power (dB)")
plt.title("Instantaneous Power vs Time")
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 600)      # time from 0 to 600 seconds
plt.show()

# Print total power
total_power = total_energy / duration
print(f"Total power (linear): {total_power:.3e}")
print(f"Total power (dB): {10*np.log10(total_power + 1e-20):.2f} dB")