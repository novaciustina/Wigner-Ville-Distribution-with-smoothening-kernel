"""
Wigner-Ville Distribution (WVD) + Spectrogram Analysis
FUNcube-1 (NORAD 39444) – DopTrack Delft – 2026-01-01

File:    FUNcube-1_39444_202601010247.fc32
Format:  complex float32 (interleaved I/Q)

Requirements:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, decimate, butter, filtfilt
import os, sys

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
FC32_FILE      = r"C:\Users\Toby\Downloads\FUNcube-1_39444_202601010247.fc32"

FS             = 25_000
TUNING_FREQ    = 145_934_500
BEACON_FREQ    = 145_935_000
DURATION       = 660

T_TCA          = (4 * 60 + 44.1)   # 284.1 s from start

# Spectrogram
NPERSEG        = 4096               # larger window → cleaner spectrogram
NOVERLAP       = NPERSEG * 7 // 8

# WVD zoom window
WVD_ZOOM_S     = 30                 # ± seconds around TCA

# Pre-filter: narrow bandpass around the beacon before WVD
# Set to None to skip, or (low_hz, high_hz) in baseband
BANDPASS_HZ    = (-800, 2500)       # beacon sweeps from ~+2kHz down to ~-500Hz

# Decimation AFTER bandpass (reduces WVD cost, keeps signal intact)
WVD_DECIMATE   = 10                 # → 2500 Hz effective rate

# SPWVD kernel sizes  (LARGER = SMOOTHER)
WVD_N_FREQ     = 256
WVD_LAG_WIN    = 120                # frequency-domain smoothing (was 100)
WVD_TIME_SMOOTH= 15                 # time-domain smoothing     (was 5)
WVD_TIME_STEP  = 5                  # columns to compute

# Post-WVD 2-D Gaussian smoothing
POST_SMOOTH_T  = 5                  # time-axis kernel half-width (WVD columns)
POST_SMOOTH_F  = 5                  # freq-axis kernel half-width (bins)

# Colormap percentile clipping for WVD panel
WVD_VMIN_PCT   = 60                 # raise floor → hide noisy background
WVD_VMAX_PCT   = 99.8

FREQ_ZOOM      = (-3000, 3000)
OUT_PNG        = "wvd_funcube1_analysis_v2.png"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def bandpass_complex(x, low_hz, high_hz, fs, order=6):
    """Bandpass a complex IQ signal by filtering I and Q separately."""
    nyq = fs / 2.0
    b, a = butter(order, [max(0.001, (low_hz + nyq) / (2 * nyq)),
                           min(0.999, (high_hz + nyq) / (2 * nyq))],
                  btype='band')
    return filtfilt(b, a, x.real).astype(np.float32) + \
      1j * filtfilt(b, a, x.imag).astype(np.float32)


def gaussian_kernel_1d(half_width):
    x = np.arange(-half_width, half_width + 1, dtype=float)
    g = np.exp(-0.5 * (x / (half_width / 2.5)) ** 2)
    return g / g.sum()


def smooth_2d(arr, ht, hf):
    """Separable 2-D Gaussian smooth along time (axis=1) and freq (axis=0)."""
    from scipy.ndimage import convolve1d
    kt = gaussian_kernel_1d(ht)
    kf = gaussian_kernel_1d(hf)
    out = convolve1d(arr, kf, axis=0, mode='reflect')
    out = convolve1d(out, kt, axis=1, mode='reflect')
    return out


def compute_spwvd(x, t_indices, n_freq, lag_window, time_smooth, fs):
    N    = len(x)
    NT   = len(t_indices)
    NFFT = 2 * n_freq
    lag_win = np.hanning(2 * lag_window + 1)
    WVD = np.zeros((NFFT, NT), dtype=float)

    for col, n in enumerate(t_indices):
        if col % max(1, NT // 20) == 0:
            pct = 100 * col / NT
            print(f"    SPWVD: {pct:5.1f}%  ({col}/{NT})", end='\r', flush=True)

        for ts in range(-time_smooth, time_smooth + 1):
            nc = int(np.clip(n + ts, 0, N - 1))
            kernel = np.zeros(NFFT, dtype=complex)
            for lag in range(-lag_window, lag_window + 1):
                n1, n2 = nc + lag, nc - lag
                if 0 <= n1 < N and 0 <= n2 < N:
                    kernel[lag % NFFT] += lag_win[lag + lag_window] * x[n1] * np.conj(x[n2])
            WVD[:, col] += np.real(np.fft.fft(kernel))

    WVD /= (2 * time_smooth + 1)
    WVD  = np.fft.fftshift(WVD, axes=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT)) * fs
    print(f"\n    SPWVD done.")
    return WVD, freqs

# ─────────────────────────────────────────────────────────────
# 1.  Load
# ─────────────────────────────────────────────────────────────
print(f"Loading: {FC32_FILE}")
if not os.path.isfile(FC32_FILE):
    sys.exit(f"[ERROR] File not found: {FC32_FILE}")

raw = np.fromfile(FC32_FILE, dtype=np.float32)
iq  = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
del raw
iq -= iq.mean()
N_TOTAL = len(iq)
print(f"  {N_TOTAL:,} complex samples  ({N_TOTAL/FS:.1f} s)")

# ─────────────────────────────────────────────────────────────
# 2.  Spectrogram (full pass)
# ─────────────────────────────────────────────────────────────
print("Computing spectrogram …")
f_spec, t_spec, Sxx = spectrogram(
    iq, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
    window='hann', return_onesided=False, detrend=False,
)
f_spec = np.fft.fftshift(f_spec)
Sxx    = np.fft.fftshift(Sxx, axes=0)
Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-30)
print(f"  Spectrogram shape: {Sxx_dB.shape}")

# ─────────────────────────────────────────────────────────────
# 3.  Extract zoom window
# ─────────────────────────────────────────────────────────────
idx_tca   = int(T_TCA * FS)
idx_start = max(0, idx_tca - WVD_ZOOM_S * FS)
idx_stop  = min(N_TOTAL, idx_tca + WVD_ZOOM_S * FS)
iq_zoom   = iq[idx_start:idx_stop].copy()
del iq
print(f"Zoom window: {idx_start/FS:.1f}–{idx_stop/FS:.1f} s  ({len(iq_zoom):,} samples)")

# ─────────────────────────────────────────────────────────────
# 4.  Narrow bandpass around beacon signal
# ─────────────────────────────────────────────────────────────
if BANDPASS_HZ is not None:
    print(f"Bandpass filtering: {BANDPASS_HZ[0]} – {BANDPASS_HZ[1]} Hz …")
    iq_filt = bandpass_complex(iq_zoom, BANDPASS_HZ[0], BANDPASS_HZ[1], FS)
else:
    iq_filt = iq_zoom

# ─────────────────────────────────────────────────────────────
# 5.  Decimate
# ─────────────────────────────────────────────────────────────
def decimate_complex(x, factor):
    """Safe complex decimation in one or two steps."""
    factors = []
    f = factor
    for p in [5, 5, 4, 4, 3, 3, 2, 2]:
        if f % p == 0:
            factors.append(p)
            f //= p
        if f == 1:
            break
    if f > 1:
        factors.append(f)
    out = x
    for p in factors:
        out = decimate(out.real, p, zero_phase=True) + \
         1j * decimate(out.imag, p, zero_phase=True)
    return out.astype(np.complex64)

print(f"Decimating ×{WVD_DECIMATE} …")
iq_ds = decimate_complex(iq_filt, WVD_DECIMATE)
FS_DS = FS / WVD_DECIMATE
print(f"  Decimated: {len(iq_ds):,} samples @ {FS_DS:.0f} Hz")

# ─────────────────────────────────────────────────────────────
# 6.  SPWVD
# ─────────────────────────────────────────────────────────────
print("Computing SPWVD …")
N_DS   = len(iq_ds)
tca_ds = int((T_TCA - idx_start / FS) * FS_DS)
w_ds   = int(WVD_ZOOM_S * FS_DS)

t_eval = np.arange(
    max(0, tca_ds - w_ds),
    min(N_DS, tca_ds + w_ds),
    WVD_TIME_STEP, dtype=int,
)
t_eval_s = t_eval / FS_DS + idx_start / FS

WVD, f_wvd = compute_spwvd(
    iq_ds, t_eval,
    n_freq=WVD_N_FREQ, lag_window=WVD_LAG_WIN,
    time_smooth=WVD_TIME_SMOOTH, fs=FS_DS,
)

# Keep positive-frequency half
half  = WVD_N_FREQ
WVD   = WVD[half:, :]
f_wvd = f_wvd[half:]

# Clip negative cross-term artefacts
WVD = np.clip(WVD, 0, None)

# ── Post-processing: 2-D Gaussian smooth ──────────────────
print("Post-smoothing WVD …")
WVD = smooth_2d(WVD, POST_SMOOTH_T, POST_SMOOTH_F)
WVD = np.clip(WVD, 0, None)

WVD_dB = 10 * np.log10(WVD + 1e-30)
print(f"  Final SPWVD shape: {WVD_dB.shape}")

# ─────────────────────────────────────────────────────────────
# 7.  Instantaneous frequency
# ─────────────────────────────────────────────────────────────
phase_ds = np.unwrap(np.angle(iq_ds))
if_raw   = np.diff(phase_ds) * FS_DS / (2 * np.pi)
t_if_s   = np.arange(len(if_raw)) / FS_DS + idx_start / FS

SMOOTH_WIN = 201
if_smooth  = np.convolve(if_raw, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode='same')

# ─────────────────────────────────────────────────────────────
# 8.  Plot
# ─────────────────────────────────────────────────────────────
print("Rendering figure …")

plt.rcParams.update({
    'font.family'     : 'DejaVu Sans',
    'font.size'       : 10,
    'axes.titlesize'  : 11,
    'figure.facecolor': '#0d1117',
    'axes.facecolor'  : '#0d1117',
    'text.color'      : '#e6edf3',
    'axes.labelcolor' : '#e6edf3',
    'xtick.color'     : '#8b949e',
    'ytick.color'     : '#8b949e',
    'axes.edgecolor'  : '#30363d',
    'axes.grid'       : True,
    'grid.color'      : '#21262d',
    'grid.linewidth'  : 0.5,
})

fig = plt.figure(figsize=(17, 12))
fig.suptitle(
    "FUNcube-1 (NORAD 39444)  ·  DopTrack Delft  ·  2026-01-01 01:47–01:58 UTC\n"
    f"Tuning: {TUNING_FREQ/1e6:.4f} MHz  |  Max elevation: 8°  |  "
    f"TCA: 01:52:44 UTC  |  fs = {FS/1e3:.0f} kHz",
    color='#58a6ff', fontsize=13, fontweight='bold', y=0.99,
)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.42, wspace=0.28,
                       left=0.07, right=0.97, top=0.92, bottom=0.07)

# ── Panel A: Spectrogram ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
vmin1 = np.percentile(Sxx_dB, 5)
vmax1 = np.percentile(Sxx_dB, 99.5)
im1 = ax1.pcolormesh(t_spec, f_spec, Sxx_dB,
                     shading='auto', cmap='viridis',
                     vmin=vmin1, vmax=vmax1, rasterized=True)
ax1.axvline(T_TCA, color='#f85149', lw=1.4, linestyle='--',
            label=f'TCA ({T_TCA:.0f} s)')
ax1.set_xlabel("Time from recording start (s)")
ax1.set_ylabel("Baseband frequency (Hz)")
ax1.set_title("STFT Spectrogram  –  Full Pass (660 s)", pad=6)
ax1.set_xlim(0, t_spec[-1])
ax1.set_ylim(*FREQ_ZOOM)
ax1.legend(loc='upper right', fontsize=9,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
cb1 = fig.colorbar(im1, ax=ax1, pad=0.01, aspect=35)
cb1.set_label("Power (dB)")

# ── Panel B: SPWVD ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
vmin2 = np.percentile(WVD_dB, WVD_VMIN_PCT)
vmax2 = np.percentile(WVD_dB, WVD_VMAX_PCT)
im2 = ax2.pcolormesh(t_eval_s, f_wvd, WVD_dB,
                     shading='auto', cmap='inferno',
                     vmin=vmin2, vmax=vmax2, rasterized=True)
ax2.axvline(T_TCA, color='#00d4ff', lw=1.4, linestyle='--', label='TCA')
ax2.set_xlabel("Time from recording start (s)")
ax2.set_ylabel("Baseband frequency (Hz)")
ax2.set_title(f"Smoothed Pseudo WVD  –  ±{WVD_ZOOM_S} s around TCA", pad=6)
ax2.set_xlim(t_eval_s[0], t_eval_s[-1])
# Zoom y-axis to bandpass region for cleaner view
ax2.set_ylim(0, min(FS_DS / 2, BANDPASS_HZ[1] + 200 if BANDPASS_HZ else FS_DS / 2))
ax2.legend(loc='upper right', fontsize=9,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
cb2 = fig.colorbar(im2, ax=ax2, pad=0.01, aspect=20)
cb2.set_label("WVD (dB)")

# ── Panel C: Instantaneous frequency ────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
mask = (t_if_s >= t_eval_s[0]) & (t_if_s <= t_eval_s[-1])
ax3.plot(t_if_s[mask], if_raw[mask],
         color='#8b949e', alpha=0.2, lw=0.5, label='Raw IF')
ax3.plot(t_if_s[mask], if_smooth[mask],
         color='#3fb950', lw=2.0, label='Smoothed IF')
ax3.axvline(T_TCA, color='#f85149', lw=1.4, linestyle='--', label='TCA')
ax3.axhline(BEACON_FREQ - TUNING_FREQ, color='#a5d6ff',
            lw=0.9, linestyle=':', alpha=0.8,
            label=f'Beacon offset ({BEACON_FREQ - TUNING_FREQ} Hz)')
ax3.set_xlabel("Time from recording start (s)")
ax3.set_ylabel("Instantaneous frequency (Hz)")
ax3.set_title("Instantaneous Frequency Estimate", pad=6)
ax3.set_xlim(t_eval_s[0], t_eval_s[-1])
ax3.set_ylim(*FREQ_ZOOM)
ax3.legend(loc='upper right', fontsize=8.5,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nFigure saved → {os.path.abspath(OUT_PNG)}")
plt.show()
print("Done.")