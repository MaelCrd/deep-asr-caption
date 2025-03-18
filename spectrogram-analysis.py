import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
import parquet_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio

import librosa

parquet_filename = "valid-00000-of-00001.parquet"
dataframe = parquet_dataframe.dataframe_from_parquet("dataset/" + parquet_filename).head(5) # Load a subset for testing
sample_rate = 16000

# wav_folder = "Datasets/LJSpeech-1.1/wavs/"
# filenames = sorted(os.listdir(wav_folder))[:5]

# spectrogram_transform = Spectrogram()

def custom_spectrogram_from_file(wav_path):
    # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
    audio, orig_sr = librosa.load(wav_path) 
    frame_length = 256 
    frame_step = 160
    fft_length = 384
    # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
    # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
    # The resulting spectrogram is also transposed for convenience
    spectrogram = librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T
    # Take the absolute value of the spectrogram to obtain the magnitude spectrum
    spectrogram = np.abs(spectrogram)
    # Take the square root of the magnitude spectrum to obtain the log spectrogram
    spectrogram = np.power(spectrogram, 0.5)
    # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
    # A small value of 1e-10 is added to the denominator to prevent division by zero.
    spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)
    return spectrogram.T

def custom_spectrogram(waveform):
    # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
    frame_length = 256 
    frame_step = 160
    fft_length = 384
    # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
    # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
    # The resulting spectrogram is also transposed for convenience
    spectrogram = librosa.stft(waveform, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T
    # Take the absolute value of the spectrogram to obtain the magnitude spectrum
    spectrogram = np.abs(spectrogram)
    # Take the square root of the magnitude spectrum to obtain the log spectrogram
    spectrogram = np.power(spectrogram, 0.5)
    # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
    # A small value of 1e-10 is added to the denominator to prevent division by zero.
    spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)
    return spectrogram.T

# Create subplot grid
fig, axs = plt.subplots(5, 1, figsize=(8, 17))

for i in range(len(dataframe)):
# for i in range(len(filenames)):
    # waveform from dataframe
    audio_bytes = dataframe.iloc[i]['audio']['bytes']
    audio_ndarray = np.array(parquet_dataframe.audio_bytes_to_ndarray(audio_bytes))
    waveform = torch.tensor(audio_ndarray).float() # Convert to tensor
    # # save waveform to file
    # waveform = torch.tensor(audio_ndarray).float().unsqueeze(0) # Convert to tensor and add a new dimension
    # torchaudio.save("test.wav", waveform, sample_rate)
    
    # # waveform from file
    # waveform, sr = torchaudio.load(wav_folder + filenames[i])
    # waveform = waveform[0] # Convert to mono

    # Apply Spectrogram transformation
    # spectrogram = spectrogram_transform(waveform)
    # spectrogram = custom_spectrogram(wav_folder + filenames[i])
    # spectrogram = custom_spectrogram("test.wav")
    spectrogram = custom_spectrogram(waveform.numpy())
    
    # # Normalize spectrogram
    # spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
    
    # Display MFCCs
    im = axs[i].imshow(spectrogram[:,:], origin='lower', aspect='auto', cmap='viridis') # Display spectrogram
    # im = axs[i].imshow(spectrogram[:,:].numpy(), origin='lower', aspect='auto', cmap='viridis') # Display spectrogram
    axs[i].set_title("Spectrogram")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Frequency")
    # axs[i].colorbar()
    
    fig.colorbar(im, ax=axs[i])  # Add color bar for reference
    
plt.show()
    
    