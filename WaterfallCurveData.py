import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from the .dat file to create the waterfall curve
data = np.loadtxt('C:\\Users\\novac\\Downloads\\FUNcube-1_39444_202601010247.dat', skiprows=1)

def update_waterfall_curve(data):
    # Use time and frequency columns from the data
    time = data[:, 0]
    freq = data[:, 1]
    baud_rate = 1200

    # Use beacon frequency to shift the frequency values, depends on satellite
    # Check other satellites for their beacon frequencies and update accordingly
    f_beacon = 145.935 * 10**6 # Hz, the frequency of the beacon signal
    freq = freq - f_beacon # Shift the frequency values 

    # Approximate the frequency curve with a function ( arctangent fit )
    def arctan_func(t, a, b, c, d):
        return a * np.arctan(b * (t - c)) + d

    # General parameters for initial guess
    def initial_guess(time, freq):
        # Amplitude of the arctan function, as the total range of arctan is - pi/2 to pi/2, so a is basically (max - min) / pi
        a = ( max(freq) - min(freq) ) / np.pi  
        # Slope of the arctan function, which can be estimated from the steepest part of the frequency curve
        b = 1 / ( 2* baud_rate ) # Try this for initial guess, as the slope of the arctan is steepest at the inflection point, and the baud rate gives an estimate of how quickly the frequency changes around that point
        # Inflection point is at t=c, where the second derivative of arctan is 0
        c = time[len(time) // 2] # Use midpoint of data for initial guess, as the inflection point is likely around the middle of the time range
        # This should be good enough for initial guess
        d = ( max(freq) + min(freq) ) / 2  # Midpoint of the frequency range
        return a, b, c, d
    a0, b0, c0, d0 = initial_guess( time, freq)
    #print("Initial guess:", a0, b0, c0, d0)
    p0 = [a0, b0, c0, d0]
    popt, pcov = curve_fit(arctan_func, time, freq, p0=p0) # returns the optimal parameters and the covariance of the parameters

    # Evaluate the fitted curve at the time points
    fitted_freq = arctan_func(time, *popt)      
    # Find inflection point of the fitted curve, which is where the second derivative changes sign
    a, b, c, d = popt
    inflection_point = c  # for arctan, the inflection point is at t=c
    #print("Inflection point in time =", inflection_point, "seconds")

    # Shift the frequency curve so that the inflection point is at 0 Hz
    freq = freq - d # Subtract the frequency value at the inflection point from all frequency values
    fitted_freq = arctan_func(time, *popt) - d # Shift the fitted curve as well to validate
    return freq, fitted_freq, time

freq, fitted_freq, time = update_waterfall_curve(data)

""""
# Plot the original frequency curve and the fitted curve
plt.figure(figsize=(10, 6))

plt.plot(freq, time, color='blue', label='Original S Curve')
plt.plot(fitted_freq, time, color='red', linewidth=2, label='Arctan fitted Curve')

plt.gca().invert_yaxis()

plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (s)')
#plt.title('Waterfall Curve and Fit')

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""
