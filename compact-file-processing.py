import pandas as pd, numpy as np, scipy.fft as fft, matplotlib.pyplot as plt, sounddevice as sd, pygame, time
from scipy.signal import find_peaks

def FileIn():
    file_path = 'c-major-chord.txt'
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows = 1)   # data from the csv file
    sounddata = df.to_numpy(dtype=float)    # Convert the data to a 2d numpy array
    sounddata[:,1] = (2*(sounddata[:,1]-np.min(sounddata[:,1]))/(np.max(sounddata[:,1])-np.min(sounddata[:,1])))-1  # Normalize the signal
    return sounddata

def Transform(signal):
    samplerate = 17312.31 # from SampleRateFinder.py
    rfftsignal = abs(fft.rfft(signal[:, 1]))    # rfft is used to get the positive half of the spectrum
    rfftf = fft.rfftfreq(len(signal[:, 1].tolist()), 1 / samplerate) # rfftfreq for frequency values
    '''
    fftsignal = fft.fft(signal[:, 1])
    fftf = fft.fftfreq(len(signal[:, 1].tolist()), 1 / samplerate)
    posfftf = fftf[:len(fftf)//2]                                   # Alternative mehtod to get the same result manually changing result to rfft from fft
    posfftsignal = np.abs(fftsignal[:len(fftsignal)//2])
    '''
    return rfftf, rfftsignal

def TriplePlot(array1, array2, array3, peaks):
    plt.figure(figsize=(18, 7))

    # Original audio input signal plot
    plt.subplot(3, 1, 1)
    plt.plot(array1[:,0], array1[:,1], linestyle='-', color='r')
    plt.title('Original Audio Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')

    # FFT plot of frequencies
    plt.subplot(3, 1, 2)
    plt.plot(array2[1:,0], array2[1:,1], linestyle='-', color='b')  # Skip the first element to avoid the DC component
    plt.plot(array2[peaks,0], array2[peaks,1], 'ro')
    plt.title('FFT of the Audio Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT')

    # Noiseless audio output signal plot
    plt.subplot(3, 1, 3)
    plt.plot(array3[:,1], array3[:,0], linestyle='-', color='g')
    plt.title('Compressed Audio Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Ampltitude')

    plt.tight_layout()
    plt.show()

def CreateCompressedAudio(fftfunction, peaks):
    compfftsignal = np.zeros_like(fftfunction)
    compfftsignal[peaks] = fftfunction[peaks]  # Keep only the peaks in the FFT signal - set everything else to zero
    compsignal = fft.irfft(compfftsignal)   # Inverse FFT to get the compressed audio signal
    return compsignal

def PlaySound(outputsignal):
    # Play the sound out loud from the speakers on the computer and compare
    extended_signal = np.tile(outputsignal, (int(np.ceil(1731231 / len(outputsignal))), 1)) # Repeat the signal to make it long enough to hear
    extended_signal = extended_signal[:,:1731231] # Trim the signal to the correct length
    sd.play(extended_signal, samplerate=17312.31)
    time.sleep(1)
    sd.stop()

def AnimateSignals(input_data, output_data, duration=10, fps=60):
    pygame.init()
    screen = pygame.display.set_mode((800, 600)) # Set screen dimensions
    pygame.display.set_caption("Output(Blue) and Input (Red) Signal Oscillation Animation:")
    clock = pygame.time.Clock() # Set up the clock for frame rate

    # Scale data to fit within screen dimensions: normalize time to screen width, amplitude to screen height
    input_data_scaled, output_data_scaled = np.copy(input_data), np.copy(output_data)
    for data, scaled_data in [(input_data, input_data_scaled), (output_data, output_data_scaled)]:
        scaled_data[:, 0] = np.interp(data[:, 0], (np.min(data[:, 0]), np.max(data[:, 0])), (0, 800))
        scaled_data[:, 1] = np.interp(data[:, 1], (np.min(data[:, 1]), np.max(data[:, 1])), (600 // 4, 3 * 600 // 4))
    
    # Frame counter for smooth animation
    num_points, frame, running = len(input_data), 0, True

    while running:
        screen.fill((0, 0, 0))  # Empty screen with black background
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False # Exit the loop
        
        current_frame = frame % num_points # Calculate current segment to plot based on the frame number
        current_input, current_output = input_data_scaled[:current_frame], output_data_scaled[:current_frame]

        # Draw input_data red & output_data blue
        for data, color in [(current_input, (255, 0, 0)), (current_output, (0, 0, 255))]:
            if len(data) > 1:
                pygame.draw.lines(screen, color, False, [(x, 600 - y) for x, y in data], 2)

        # Update the display, limit frame rate, increment frame for animation
        pygame.display.flip()
        clock.tick(fps)
        frame += 1
        if frame >= num_points * duration:
            running = False
    pygame.quit()

input_data = FileIn()
fft_frequency, fft_signal = Transform(input_data)
fft_data = np.column_stack((fft_frequency, fft_signal))
peaks, _ = find_peaks(fft_data[:,1], height = 5)

noiseless_data = CreateCompressedAudio(fft_data[:,1], peaks)

print(f"The peak frequencies: {fft_data[peaks,0]}")

output_data = np.column_stack((noiseless_data, input_data[:-1,0]))
TriplePlot(input_data, fft_data, output_data, peaks)
PlaySound(output_data)
PlaySound(input_data)

AnimateSignals(input_data, output_data[:, ::-1])
