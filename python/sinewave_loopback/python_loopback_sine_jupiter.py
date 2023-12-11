import numpy as np
import adi
import matplotlib.pyplot as plt
import time

############################################################################################################
# Configure SDR ############################################################################################
############################################################################################################
sdr = adi.adrv9002(uri="ip:10.48.65.212") # Create Radio

frequency = 15360  # 15.360 kHz sinewave to be transmitted
amplitude = 2**14
center_freq = 915000000 # Hz
sample_rate = sdr.rx0_sample_rate
print("sample rate: ", sample_rate)
num_samps = int((20*sample_rate)/frequency) # number of samples per call to rx()

sdr.rx_ensm_mode_chan0 = "rf_enabled"
sdr.rx_ensm_mode_chan1 = "rf_enabled"
sdr.tx_hardwaregain_chan0 = -5
sdr.rx_hardwaregain_chan0 = 5
sdr.tx_ensm_mode_chan0 = "rf_enabled"
sdr.tx_cyclic_buffer = True
sdr._tx_buffer_size = num_samps # number of samples per call to tx()
sdr.rx_buffer_size = num_samps
sdr.tx0_lo = center_freq
sdr.rx0_lo = center_freq
############################################################################################################
############################################################################################################

############################################################################################################
# Create and plot a complex sinusoid #######################################################################
############################################################################################################
# Calculate time values
t = np.arange(num_samps) / sample_rate
# Generate sinusoidal waveform
phase_shift = -np.pi/2  # Shift by -90 degrees
tx_samples = amplitude * (np.cos(2 * np.pi * frequency * t + phase_shift) + 1j*np.sin(2 * np.pi * frequency * t + phase_shift))

# Plot Tx time domain
plt.figure(1)
plt.plot(t, np.real(tx_samples), label = "I (Real)")
plt.plot(t, np.imag(tx_samples), label = "Q (Imag)")
plt.legend()
plt.title('Tx time domain')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Calculate Tx spectrum in dBFS
tx_samples_fft = tx_samples * np.hanning(num_samps)
ampl_tx = (np.abs(np.fft.fftshift(np.fft.fft(tx_samples_fft))))
fft_txvals_iq_dbFS = 10*np.log10(np.real(ampl_tx)**2 + np.imag(ampl_tx)**2) + 20*np.log10(2/2**(16-1))\
                                         - 20*np.log10(len(ampl_tx))
f = np.linspace(sample_rate/-2, sample_rate/2, len(fft_txvals_iq_dbFS))

# Plot Tx freq domain
plt.figure(2)
plt.plot(f/1e6, fft_txvals_iq_dbFS)
plt.xlabel("Frequency [MHz]")
plt.ylabel("dBFS")
plt.title('Tx FFT')

# Constellation plot for the transmit data
plt.figure(3)
plt.plot(np.real(tx_samples), np.imag(tx_samples), '.')
plt.xlabel("I (Real) Sample Value")
plt.ylabel("Q (Imag) Sample Value")
plt.grid(True)
plt.title('Constellation Plot Tx')
############################################################################################################
############################################################################################################

############################################################################################################
# Call Tx function to start transmission ###################################################################
############################################################################################################
sdr.tx(tx_samples) # start transmitting
############################################################################################################
############################################################################################################

time.sleep(1) # wait for internal calibrations
# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

############################################################################################################
# Call Rx function to receive transmission and plot the data################################################
############################################################################################################
# Receive samples
rx_samples = sdr.rx()

# Stop transmitting
sdr.tx_destroy_buffer()

# Time values
t = np.arange(num_samps) / sample_rate

# Plot Rx time domain
plt.figure(4)
plt.plot(np.real(rx_samples), label = "I (Real)")
plt.plot(np.imag(rx_samples), label = "Q (Real)")
plt.legend()
plt.title('Rx time domain')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Calculate Rx spectrum in dBFS
rx_samples_fft = rx_samples * np.hanning(num_samps)
ampl_rx = (np.abs(np.fft.fftshift(np.fft.fft(rx_samples_fft))))
fft_rxvals_iq_dbFS = 10*np.log10(np.real(ampl_rx)**2 + np.imag(ampl_rx)**2) + 20*np.log10(2/2**(16-1))\
                                         - 20*np.log10(len(ampl_rx))
f = np.linspace(sample_rate/-2, sample_rate/2, len(fft_rxvals_iq_dbFS))

# Plot Rx freq domain
plt.figure(5)
plt.plot(f/1e6, fft_rxvals_iq_dbFS)
plt.xlabel("Frequency [MHz]")
plt.ylabel("dBFS")
plt.title('Rx FFT')

# Constellation plot for the transmit data
plt.figure(6)
plt.plot(np.real(rx_samples), np.imag(rx_samples), '.')
plt.xlabel("I (Real) Sample Value")
plt.ylabel("Q (Imag) Sample Value")
plt.grid(True)
plt.title('Constellation Plot Rx')
plt.show()
############################################################################################################
############################################################################################################
