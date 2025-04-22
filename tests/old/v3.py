import parquet_dataframe
import pandas as pd

dataframe = parquet_dataframe.dataframe_from_parquet("valid-00000-of-00001.parquet")
dataframe = dataframe.head(10)
for i in range(0, len(dataframe)):
    dataframe['audio'][i] = parquet_dataframe.audio_bytes_to_ndarray(dataframe['audio'][i]['bytes'])

print(dataframe.head())

# with open('audio0.wav', 'wb') as f:
#     f.write(dataframe['audio'][0]['bytes'])

# quit()

import torchaudio
import torch

# filepath = "audio0.wav"  # Replace with your audio file path
# waveform, sample_rate = torchaudio.load(filepath)

sample_rate = 16000
waveform = torch.tensor([dataframe['audio'][1]]).float()

print("Waveform shape:", waveform.shape)  # [channels, time_steps] (usually 1 channel for mono)
print("Sample rate:", sample_rate)      # Samples per second

import matplotlib.pyplot as plt

plt.figure()
plt.plot(waveform.t().numpy())  # Transpose to [time_steps, channels] and convert to NumPy
plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


from torchaudio.transforms import MFCC
import torch

n_mfcc = 10  # Number of MFCC coefficients to compute
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc) # Initialize MFCC transform

mfccs = mfcc_transform(waveform) # Apply transform to your waveform

print("MFCCs shape:", mfccs.shape) # [channels, n_mfcc, time_frames]

plt.figure()
plt.imshow(mfccs[0,:,:].numpy(), origin='lower', aspect='auto') # Display first channel's MFCCs
plt.title("MFCCs")
plt.xlabel("Time Frames")
plt.ylabel("MFCC Coefficients")
plt.colorbar()
plt.show()