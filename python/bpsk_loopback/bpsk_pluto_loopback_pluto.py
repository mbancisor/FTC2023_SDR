import time

import adi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

############################################################################################################
# Configure SDR ############################################################################################
############################################################################################################
# Configure properties
sdr = adi.Pluto("ip:192.168.2.1") # Create radio
sdr.rx_rf_bandwidth = 4000000
sdr.rx_lo = 914950000
sdr.rx_buffer_size = 32768
sdr.sample_rate = 2000000
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.rx_hardwaregain_chan0 = 0

sdr.tx_lo = 915000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -25
sdr._tx_buffer_size = 1024 # 32 times smaller than rx buffer

fs = int(sdr.sample_rate)
############################################################################################################
############################################################################################################

############################################################################################################
# Create array of bits #####################################################################################
############################################################################################################
num_symbols = 64 # 64 * 32 = 2048 symbols will be received
sps = 16
bits = np.random.randint(0, 2, num_symbols) # Our data to be transmitted, 1's and 0'
bits_transmitted = bits #save bits for plots
############################################################################################################
############################################################################################################

############################################################################################################
# Interpolate with 16sps and Remap symbols: bit 0 -> -1; bit 1 -> 1 ########################################
############################################################################################################
pulse_train = np.array([])
for bit in bits:
    pulse = np.ones(sps)*(bit*2-1)
    pulse[0] = bit*2-1 # set the first value to either a 1 or -1
    pulse_train = np.concatenate((pulse_train, pulse)) # add the 16 samples to the signal

# Make the data complex
samples_tx = pulse_train.astype(complex)
samples_tx = samples_tx * 2**14
############################################################################################################
############################################################################################################

############################################################################################################
# Call tx() function and transmit ##########################################################################
############################################################################################################
sdr.tx_destroy_buffer()
time.sleep(1)
sdr.tx(samples_tx)
time.sleep(3)
############################################################################################################
############################################################################################################

# Collect data
for r in range(1):

    ############################################################################################################
    # Call rx() function to receive data #######################################################################
    ############################################################################################################
    for i in range (0, 10):
        samples = sdr.rx()
    time.sleep(1)    
    samples = sdr.rx()    
        #time.sleep(1)
    # Remap received samples between -1 and 1
    max_real = np.amax(abs(np.real(samples)))
    max_imag = np.amax(abs(np.imag(samples)))
    max_val = max(max_real, max_imag)
    samples *= (1/max_val)
    print(samples)

    # Save receive samples for plots
    samples_rx_raw = samples
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    # Adjust frequency offset ##################################################################################
    ############################################################################################################   
    # Coarse frequancy adjustment
    samples_adjust = samples**2 # square the received signal to obtain 2* offset frequency
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_adjust)))
    f = np.linspace(-fs/2.0, fs/2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    Ts = 1/fs # calc sample period
    t = np.arange(0, Ts*len(samples), Ts) # create time vector
    samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)

    # Save samples after coarse frequency adjustment for plots
    samples_rx_coarse_freq_adj = samples
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    # Select only the right samples and decimate ###############################################################
    ############################################################################################################  
    samples_interpolated = signal.resample_poly(samples, 16, 1) # interpolation
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=complex)
    out_rail = np.zeros(len(samples) + 10, dtype=complex) # stores values, each iteration we need the previous 
                                                          # 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    while i_out < len(samples) and i_in+16 < len(samples):
        #out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index
    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on
    # Save samples after time adjustment with interpolation for plots
    samples_rx_time_adj = samples
    ############################################################################################################
    ############################################################################################################
    
    ############################################################################################################
    # Fine frequency and phase adjustment (Costas Loop) ########################################################
    ############################################################################################################ 
    N = len(samples)
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.132
    beta = 0.00932
    out = np.zeros(N, dtype=complex)
    freq_log = []

    for repeat in range(1):
        for i in range(N):
            out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
            error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

            # # Debugging output
            # print(f"Iteration {i}: Phase={phase}, Freq={freq}, Error={error}")

            # Advance the loop (recalc phase and freq offset)
            freq += (beta * error)
            freq_log.append(freq * fs / (2*np.pi)) # convert from angular velocity to Hz for logging
            phase += freq + (alpha * error)

            # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
            while phase >= 2*np.pi:
                phase -= 2*np.pi
            while phase < 0:
                phase += 2*np.pi
    samples = out

    # Save the samples after costas loop (fine frequency and phase adjustment) for plots
    samples_rx_costas_loop = samples
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    # Plot data ################################################################################################
    ############################################################################################################ 
    # Plot transmitted bits
    plt.figure(1)
    plt.plot(bits_transmitted,'.-')
    plt.xlabel("No. Bit")
    plt.ylabel("Bit Value")
    plt.grid()
    plt.xticks(np.arange(0, len(bits_transmitted), 1.0))
    plt.title('Bits transmitted')

    # Plot I and Q transmitted samples after interpolating 16 sps (Q data is 0 for BPSK)
    plt.figure(2)
    plt.plot(np.real(samples_tx),'.-', label = "I (Real)")
    plt.plot(np.imag(samples_tx),'.-', label = "Q (Imag)")
    plt.legend()
    plt.xlabel("No. Sample")
    plt.ylabel("Sample value")
    plt.grid()
    plt.xticks(np.arange(0, len(samples_tx), 16))
    plt.yticks(np.arange(-16384, +16385, 16384))
    plt.title('Samples transmitted repeat x16 (16 sps)')

    # Optional: plot the constellation on the samples transmitted
    plt.figure(3)
    plt.plot(np.real(samples_tx), np.imag(samples_tx), 'o')
    plt.xlabel("I (Real) Sample Value")
    plt.ylabel("Q (Imag) Sample Value")
    plt.grid(True)
    plt.title('Constellation Plot Tx')

    # Plot received I and Q raw data received
    plt.figure(4)
    plt.plot(np.real(samples_rx_raw),'.-', label = "I (Real)")
    plt.plot(np.imag(samples_rx_raw),'.-', label = "Q (Imag)")
    plt.legend()
    plt.xlabel("No. Sample")
    plt.ylabel("Sample value")
    plt.grid()
    plt.xticks(np.arange(0, len(samples_rx_raw), 16))
    plt.title('Samples received Raw')

    # Constellation plot for the received raw complex samples
    plt.figure(5)
    plt.plot(np.real(samples_rx_raw), np.imag(samples_rx_raw), '.')
    plt.xlabel("I (Real) Sample Value")
    plt.ylabel("Q (Imag) Sample Value")
    plt.grid(True)
    plt.title('Constellation Plot Rx Raw')

    # Plot received I and Q data vs time after coarse frequency adjustment
    samples_rx_coarse_freq_adj
    plt.figure(6)
    plt.plot(np.real(samples_rx_coarse_freq_adj),'.-', label = "I (Real)")
    plt.plot(np.imag(samples_rx_coarse_freq_adj),'.-', label = "Q (Imag)")
    plt.legend()
    plt.xlabel("No. Sample")
    plt.ylabel("Sample value")
    plt.grid()
    plt.xticks(np.arange(0, len(samples_rx_coarse_freq_adj), 16))
    plt.title('Samples received after coarse frequency adjustment')

    # Plot constellation data after coarse frequency adjustment
    plt.figure(7)
    plt.plot(np.real(samples_rx_coarse_freq_adj), np.imag(samples_rx_coarse_freq_adj), '.')
    plt.xlabel("I (Real) Sample Value")
    plt.ylabel("Q (Imag) Sample Value")
    plt.grid(True)
    plt.title('Constellation Plot Rx Coarse Freq Adj.')

    # Plot received I and Q data after time ajustment with interpolation
    plt.figure(8)
    plt.plot(np.real(samples_rx_time_adj),'.-', label = "I (Real)")
    plt.plot(np.imag(samples_rx_time_adj),'.-', label = "Q (Imag)")
    plt.legend()
    plt.xlabel("No. Sample")
    plt.ylabel("Sample value")
    plt.grid()
    plt.xticks(np.arange(0, len(samples_rx_time_adj), 16))
    plt.title('Samples received after selecting the right samples (1 of each 16)')

    # Constellation plot for the received data after time adjustment with interpolation 
    plt.figure(9)
    plt.plot(np.real(samples_rx_time_adj), np.imag(samples_rx_time_adj), '.')
    plt.xlabel("I (Real) Sample Value")
    plt.ylabel("Q (Imag) Sample Value")
    plt.grid(True)
    plt.title('Constellation Plot Rx After selecting the right samples')

    # Plot freq over time the frequency error of the costats loop to see how long it takes to hit the right offset
    plt.figure(10)
    plt.plot(freq_log,'.-')
    plt.title('Frequency error vs time of Costas Loop')

    # Plot received I and Q data after costas loop (fine frequency and phase adjustment)
    plt.figure(11)
    plt.plot(np.real(samples_rx_costas_loop),'.-', label = "I (Real)")
    plt.plot(np.imag(samples_rx_costas_loop),'.-', label = "Q (Imag)")
    plt.legend()
    plt.xlabel("No. Sample")
    plt.ylabel("Sample value")
    plt.grid()
    plt.xticks(np.arange(0, len(samples_rx_costas_loop), 16))
    plt.title('Samples received after costas loop')

    # Constellation plot for the received data after costas loop (fine frequency and phase adjustment)
    plt.figure(12)
    plt.plot(np.real(samples_rx_costas_loop), np.imag(samples_rx_costas_loop), '.', label = "all samples")
    plt.xlabel("I (Real) Sample Value")
    plt.ylabel("Q (Imag) Sample Value")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.grid(True)
    plt.title('Constellation Plot Rx Costas Loop')
    # eliminate fist 200 samples to pass the costas loop adjustment 
    samples_rx_costas_loop = samples_rx_costas_loop[200:]
    plt.plot(np.real(samples_rx_costas_loop), np.imag(samples_rx_costas_loop), '.', label = "without first 200 samples")
    plt.legend()
    ############################################################################################################
    ############################################################################################################

plt.show()
