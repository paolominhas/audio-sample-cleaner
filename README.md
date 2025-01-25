## datahandlingproject2024
# What this code does:
This code processes audio data:
- Takes audio input from Arduino
- This is then processed using a Raspberry Pi in python to make a csv file of the waveform
- This waveform is then normalised and an FFT is applied to take the wave into k-space
- The peaks are identified using scipy.findpeaks
- The other noise is removed (set to zero)
- An inverse FFT is applied to get a wave made up only of the main frequencies
- The sound is played

This code was originally for a university course but is now spiced up with some more interesting features.
