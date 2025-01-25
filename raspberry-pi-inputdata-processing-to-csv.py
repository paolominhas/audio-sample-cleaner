import serial
import matplotlib.pyplot as plt
import numpy as np
import time

'''
L = 1
N = 1000

mesh = np.linspace(0, L, N, endpoint = False)
f = lambda x: np.sin(2 * x)
f_h = f(mesh)

f_fft_h = np.fft.fft(f_h)

plt.plot(mesh, f_h , label= "Fucntional sine or something")
plt.plot(mesh, f_fft_h , label= "Fucntional sine or something fourier transform")
plt.legend()
plt.show()
'''

import serial
import matplotlib.pyplot as plt
import numpy as np
import time

f= 'gensine4402.5v'
# Set the serial port and baud rate
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=5)

# Give the serial connection some time to establish
time.sleep(1)
ser.flushInput()
time.sleep(1)

# Send a command to Arduino to start data transmission
ser.write(bytes([29, 2]))
time.sleep(1)

# Start timing for data collection
start = time.time()
data = ser.read(1024)  # Read 1024 bytes of data

end = time.time()

# Convert byte data to numpy array (assuming 8-bit data)
values = np.frombuffer(data, dtype=np.uint8)

# If you're using 16-bit data, use np.int16 instead

# Remove the first value if necessary (e.g., header byte)
values = np.delete(values, 0)

T = end - start
# Define sampling rate (e.g., 8 kHz)
sampling_rate = 17310
#(len(values))/(T*781)
#17000
#((len(values))/(T*1000))
#16000

print((len(values))/(T*781))

# Calculate the FFT
fft_vals = np.fft.fft(values)

# Get the corresponding frequencies
freq_bins = np.fft.fftfreq(len(values), 1/sampling_rate)


# Only take the positive half of the FFT result
positive_freqs = freq_bins[:len(freq_bins)//2]
positive_fft_vals = np.abs(fft_vals)[:len(fft_vals)//2]

values_data = np.column_stack((np.arange(len(values))/sampling_rate, values))
np.savetxt(f'time_domain_signal{f}.txt', values_data, header="Time-domain signal values (raw microphone data)", comments='')
# Save the FFT frequency and magnitude values to a text file
fft_data = np.column_stack((positive_freqs, positive_fft_vals))
np.savetxt(f'fft_data{f}.txt', fft_data, header="Frequency (Hz), Magnitude (FFT)", comments='', delimiter=',')

#np.arange(len(values))/sampling_rate
t = np.linspace(0, end-start, len(values))
# Plot the time domain signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(values))/sampling_rate, values)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the frequency domain (FFT)
plt.subplot(2, 1, 2)
plt.plot(positive_freqs[10:], positive_fft_vals[10:])
plt.title('Frequency Domain (FFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')


plt.tight_layout()
plt.savefig(f'signal_plots{f}.png')  # Save the plots to an image file
plt. import serial
import matplotlib.pyplot as plt
import numpy as np
import time

'''
L = 1
N = 1000

mesh = np.linspace(0, L, N, endpoint = False)
f = lambda x: np.sin(2 * x)
f_h = f(mesh)

f_fft_h = np.fft.fft(f_h)

plt.plot(mesh, f_h , label= "Fucntional sine or something")
plt.plot(mesh, f_fft_h , label= "Fucntional sine or something fourier transform")
plt.legend()
plt.show()
'''

import serial
import matplotlib.pyplot as plt
import numpy as np
import time

f= 'gensine4402.5v'
# Set the serial port and baud rate
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=5)

# Give the serial connection some time to establish
time.sleep(1)
ser.flushInput()
time.sleep(1)

# Send a command to Arduino to start data transmission
ser.write(bytes([29, 2]))
time.sleep(1)

# Start timing for data collection
start = time.time()
data = ser.read(1024)  # Read 1024 bytes of data

end = time.time()

# Convert byte data to numpy array (assuming 8-bit data)
values = np.frombuffer(data, dtype=np.uint8)

# If you're using 16-bit data, use np.int16 instead

# Remove the first value if necessary (e.g., header byte)
values = np.delete(values, 0)

T = end - start
# Define sampling rate (e.g., 8 kHz)
sampling_rate = 17310
#(len(values))/(T*781)
#17000
#((len(values))/(T*1000))
#16000

print((len(values))/(T*781))

# Calculate the FFT
fft_vals = np.fft.fft(values)

# Get the corresponding frequencies
freq_bins = np.fft.fftfreq(len(values), 1/sampling_rate)


# Only take the positive half of the FFT result
positive_freqs = freq_bins[:len(freq_bins)//2]
positive_fft_vals = np.abs(fft_vals)[:len(fft_vals)//2]

values_data = np.column_stack((np.arange(len(values))/sampling_rate, values))
np.savetxt(f'time_domain_signal{f}.txt', values_data, header="Time-domain signal values (raw microphone data)", comments='')
# Save the FFT frequency and magnitude values to a text file
fft_data = np.column_stack((positive_freqs, positive_fft_vals))
np.savetxt(f'fft_data{f}.txt', fft_data, header="Frequency (Hz), Magnitude (FFT)", comments='', delimiter=',')

#np.arange(len(values))/sampling_rate
t = np.linspace(0, end-start, len(values))
# Plot the time domain signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(values))/sampling_rate, values)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the frequency domain (FFT)
plt.subplot(2, 1, 2)
plt.plot(positive_freqs[10:], positive_fft_vals[10:])
plt.title('Frequency Domain (FFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')


plt.tight_layout()
plt.savefig(f'signal_plots{f}.png')  # Save the plots to an image file
plt.show()
\
