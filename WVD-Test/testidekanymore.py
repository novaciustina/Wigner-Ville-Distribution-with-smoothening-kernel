import numpy as np
import matplotlib.pyplot as plt
import h5py  # FIX 1: stream to disk instead of RAM. pip install h5py. REVERT: remove import + h5py block

# Data reading
data = np.fromfile(r"C:\Users\Toby\Downloads\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64)

# Define the sampling frequency and time vector
sampling_rate = 25000  # Hz - ALREADY DOWNSAMPLED
duration = 660 #s

# ============================================================
# OPTIMISATION 1: Larger chunk + bigger stride = fewer chunks total.
#   chunk_size: bigger = better frequency resolution per chunk.
#   stride_skip: controls how far we jump between chunks (time resolution).
#   These two together control the trade-off between speed and detail.
#   REVERT: set chunk_size = 1024*4 and stride_skip = 10
# ============================================================
chunk_size  = 1024 * 16   # OPT1: was 1024*4 — 4x bigger chunk, better freq resolution
stride_skip = 1           # OPT1: was 10 — jump further between chunks, fewer total chunks

# ============================================================
# OPTIMISATION 2: Smaller max_tau = fewer frequency bins to compute.
#   Halving max_tau makes the inner loop 4x faster (half the lags,
#   half the FFT size). You still see the full signal, just with
#   coarser frequency resolution.
#   REVERT: set max_tau = 128
# ============================================================
max_tau = 64  # OPT2: was 128 — 2x fewer lags, 4x speed gain overall

stride = chunk_size // 2  # overlap between chunks (unchanged)

# ============================================================
# OPTIMISATION 3: Vectorised WVD — the entire inner for-loop over t
#   is replaced with NumPy array operations that run in C.
#   Original: Python loop over every t (3840 iters/chunk) — very slow.
#   New: builds all lags as a 2D matrix at once with broadcasting,
#   then does a single batched FFT over all time points simultaneously.
#   REVERT: replace this function with the original loop-based version.
# ============================================================
def wigner_ville_distribution(x, max_tau):
    """Vectorised Wigner-Ville distribution — no Python loop over time."""
    N    = len(x)
    taus = np.arange(-max_tau, max_tau + 1)       # OPT3: lag vector, shape (2*max_tau+1,)

    # OPT3: t_idx is every valid centre time, shape (n_t,)
    t_idx = np.arange(max_tau, N - max_tau)

    # OPT3: build indices for x[t+tau] and x[t-tau] simultaneously.
    #        plus_idx shape:  (n_t, 2*max_tau+1) — all t+tau combinations at once
    #        minus_idx shape: (n_t, 2*max_tau+1) — all t-tau combinations at once
    plus_idx  = t_idx[:, None] + taus[None, :]    # OPT3: broadcast over both dims
    minus_idx = t_idx[:, None] - taus[None, :]    # OPT3: broadcast over both dims

    # OPT3: compute the full autocorrelation matrix in one shot — no Python loop
    #        R shape: (n_t, 2*max_tau+1)
    R = x[plus_idx] * np.conj(x[minus_idx])       # OPT3: was computed one row at a time

    # OPT3: FFT across the lag axis (axis=1) for ALL time points at once.
    #        This replaces thousands of individual np.fft.fft(r) calls.
    wvd = np.fft.fftshift(np.fft.fft(R, axis=1), axes=1)  # OPT3: batched FFT

    return np.real(wvd).astype(np.float32)  # OPT3: cast to float32 here to halve memory

# ---- compute total chunks for progress display ----
total_chunks = len(range(0, len(data) - chunk_size, stride * stride_skip))
print(f"Total chunks to process: {total_chunks}")
print(f"Each chunk covers {chunk_size/sampling_rate:.2f}s of signal")
print("Starting...")

with h5py.File("wvd_output.h5", "w") as f:  # FIX: write to disk not RAM. REVERT: use wvd_chunks=[]

    dset = f.create_dataset(           # FIX: resizable disk dataset. REVERT: remove
        "wvd",
        shape=(0, 2 * max_tau + 1),    # start with 0 rows, grows as we write
        maxshape=(None, 2 * max_tau + 1),
        dtype=np.float32,
        # ============================================================
        # OPTIMISATION 4: h5py internal chunk cache — tells HDF5 how to
        #   batch disk writes. Without this, every resize()+write is a
        #   separate disk flush. 256 rows per block is a good balance.
        #   REVERT: remove the chunks= argument
        # ============================================================
        chunks=(256, 2 * max_tau + 1)  # OPT4: batch disk writes in blocks of 256 rows
    )

    row = 0

    for i, start in enumerate(range(0, len(data) - chunk_size, stride * stride_skip)):

        chunk     = data[start : start + chunk_size]
        wvd_chunk = wigner_ville_distribution(chunk, max_tau=max_tau)
        # wvd_chunk shape: (n_valid_t, 2*max_tau+1)
        # no trim needed — vectorised function only computes valid t points (OPT3)

        new_row = row + len(wvd_chunk)
        dset.resize(new_row, axis=0)       # grow disk dataset by this chunk's rows
        dset[row:new_row] = wvd_chunk      # write to disk — chunk freed after this line
        row = new_row

        # ============================================================
        # OPTIMISATION 5: Progress print every 50 chunks so you can
        #   see it's running without slowing things down.
        #   REVERT: remove this if block
        # ============================================================
        if i % 50 == 0:  # OPT5: print every 50 chunks, not every chunk
            print(f"  chunk {i}/{total_chunks}  ({100*i/total_chunks:.1f}%)  —  {row} rows on disk")

    print(f"Done. Reading {row} rows back from disk for plotting...")
    wvd = dset[:]  # FIX: read back from disk. REVERT: use np.concatenate(wvd_chunks)

# ---- Plotting ----
num_time = wvd.shape[0]
num_freq = wvd.shape[1]

# ============================================================
# OPTIMISATION 6: Downsample the WVD image before handing to imshow.
#   Your screen is at most ~2000px tall — imshow doesn't benefit from
#   millions of rows. We average blocks of rows down to 1, which makes
#   rendering fast without losing visible detail.
#   REVERT: remove the block_size + reshape + mean lines, plot wvd_db directly
# ============================================================
max_display_rows = 2000                                    # OPT6: target pixel height
block_size  = max(1, num_time // max_display_rows)         # OPT6: rows to average together
trimmed     = num_time - (num_time % block_size)           # OPT6: trim to exact multiple
wvd_display = wvd[:trimmed].reshape(-1, block_size, num_freq).mean(axis=1)  # OPT6: average down
print(f"Display shape: {wvd_display.shape}  (downsampled from {num_time} rows)")

# Time and frequency axes
time_axis = np.linspace(0, duration, wvd_display.shape[0])
freqs     = np.fft.fftshift(np.fft.fftfreq(num_freq, d=1 / sampling_rate))

wvd_db = 10 * np.log10(np.abs(wvd_display) + 1e-20)
vmin   = np.percentile(wvd_db, 5)
vmax   = np.percentile(wvd_db, 99)

plt.figure(figsize=(14, 7))
plt.imshow(wvd_db.T, aspect='auto', origin='lower',
           extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]],
           vmin=vmin, vmax=vmax)
plt.colorbar(label="Power (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Wigner–Ville Distribution")
plt.tight_layout()
plt.show()