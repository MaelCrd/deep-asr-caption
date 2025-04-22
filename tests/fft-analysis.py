import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
import parquet_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parquet_filename = "valid-00000-of-00001.parquet"
dataframe = parquet_dataframe.dataframe_from_parquet(parquet_filename).head(10)  # Load a subset for testing

sample_rate = 16000
n_fft = 512  # Number of FFT points
hop_length = 256  # Overlap between windows

# Initialize STFT transform
spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="reflect", power=None)

# Create subplot grid
fig, axs = plt.subplots(1, 10, figsize=(8, 17))

for i in range(len(dataframe)):
    audio_bytes = dataframe.iloc[i]['audio']['bytes']
    audio_ndarray = np.array(parquet_dataframe.audio_bytes_to_ndarray(audio_bytes))
    waveform = torch.tensor(audio_ndarray).float()  # Convert to tensor

    # Compute STFT
    spectrogram = spectrogram_transform(waveform)  # Apply STFT transform

    # Convert to magnitude spectrogram
    magnitude_spectrogram = spectrogram.abs()

    # Calculate frequency values
    frequencies = np.linspace(0, sample_rate / 2, magnitude_spectrogram.shape[0])

    # Display magnitude spectrogram
    im = axs[i].imshow(magnitude_spectrogram[:, :].numpy(), origin='lower', aspect='auto', cmap='viridis',
                       extent=[0, magnitude_spectrogram.shape[1], frequencies[0], frequencies[-1]])
    axs[i].set_title("Magnitude Spectrogram")
    axs[i].set_xlabel("Time Frames")
    axs[i].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axs[i])  # Add color bar for reference

plt.tight_layout()
plt.show()
