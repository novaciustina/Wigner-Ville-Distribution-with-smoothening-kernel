import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg, ndimage

# Data reading
data = np.fromfile(r"C:\Users\novac\OneDrive\Desktop\Y2 Books\Q3\Project Q3\Data\FUNcube-1_39444_202601010247.fc32", dtype=np.complex64)
# Define the sampling frequency and time vector
duration = 660         # seconds
n_samples = 16500000   # total number of samples
sampling_rate = 25000  # sampling frequency in Hz - ALREADY DOWNSAMPLED

def wigner_ville_distribution(x):
    #Ensure the input is a numpy array - CHECK AGAIN IF NECCESARY I AM A BIT CONFUSED
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    # Compute the autocorrelation function matrix
    if x.ndim != 1:
        raise ValueError("Input data should be one dimensional time series.")
    """
    #Signal is already complex, so no need for Hilbert transform, but I wanted to check to see how it looks for a sinusoidal signal
    if use_analytic:
        if all(numpy.isreal(x)):
            x = signal.hilbert(x)
        else:
            raise RuntimeError("Keyword 'use_analytic' set to True but signal"
                               " is of complex data type. The analytic signal"
                               " can only be computed if the input signal is"
                               " real valued.")
    """


    # Calculate the wigner distribution
    N = len(x) #Number of samples in the input signal
    bins = np.arange(N)

    indices = linalg.hankel(bins, bins + N - (N % 2)) # Create a Hankel matrix of indices for the Wigner distribution calculation, The Hanket matrix is used as a lag structure needed for the autocorrelatio

    # Pad the input signal with zeros to handle edge effects in the Wigner distribution calculation. Padding prevents index errors.
    padded_x = np.pad(x, (N, N), 'constant')

    # The instantaneous autocorrelation
    wigner_integrand = \
        padded_x[indices+N] * np.conjugate(padded_x[indices[::, ::-1]])

    wigner_distribution = np.real(np.fft.fft(wigner_integrand, axis=1)).T
    return wigner_distribution


# Compute the Wigner-Ville distribution for the entire signal in chunks to manage memory usage
chunk_size = 2048
stride = 1024  # 50% overlap
wvd_chunks = []


for start in range(0, len(data)-chunk_size, stride):
    chunk = data[start:start+chunk_size]
    wvd_chunk = wigner_ville_distribution(chunk)
    wvd_chunks.append(wvd_chunk)
wvd = np.concatenate(wvd_chunks, axis=0)


# Center frequencies
wvd = np.fft.fftshift(wvd, axes=0)

#Visualization
plt.imshow(10*np.log10(np.abs(wvd)+1e-20), aspect='auto', origin='lower') # USE LOG SCALE FOR BETTER VISUALIZATION, ADD A SMALL CONSTANT TO AVOID LOG(0) ISSUES
plt.title("Wigner-Ville Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show() 






"""
#SANITY CHECK - Compare with spectrogram / THIS THE SAME AS WVD, JUST TO CHECK IF THE SIGNAL IS LOADED PROPERLY
f, t, Sxx = signal.spectrogram(data[0:n_samples], fs=25000)

plt.pcolormesh(t, f, 10*np.log10(Sxx))
plt.title("Spectrogram")
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.show()
"""
