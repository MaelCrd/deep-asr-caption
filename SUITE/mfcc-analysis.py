import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchaudio

# parquet_filename = "valid-00000-of-00001.parquet"
# dataframe = parquet_dataframe.dataframe_from_parquet(parquet_filename).head(5) # Load a subset for testing

sample_rate = 48000
n_mfcc = 30  # Number of MFCC coefficients to compute
# melkwargs = {
#     'n_mels': 26,  # Number of Mel filter banks
#     'n_fft': 512,  # Number of FFT points
# }
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)#, melkwargs=melkwargs) # Initialize MFCC transform

# Create subplot grid
fig, axs = plt.subplots(5, 1, figsize=(8, 17))

audio_files = [f'audio({i})_cm.mp3' for i in range(5)]

for i in range(len(audio_files)):
    # audio_bytes = dataframe.iloc[i]['audio']['bytes']
    # audio_ndarray = np.array(parquet_dataframe.audio_bytes_to_ndarray(audio_bytes))
    # waveform = torch.tensor(audio_ndarray).float() # Convert to tensor
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_files[i])
    waveform = waveform[0] # Use only the first channel
    
    print(f"Sample Rate: {sample_rate}")

    mfccs = mfcc_transform(waveform) # Apply MFCC transform
    
    # Display MFCCs
    im = axs[i].imshow(mfccs[:,:].numpy(), origin='lower', aspect='auto', cmap='viridis', interpolation='none') # Display first channel's MFCCs
    # axs[i].imshow(mfccs[:,:].numpy(), origin='lower', aspect='auto') # Display first channel's MFCCs
    axs[i].set_title("MFCCs")
    axs[i].set_xlabel("Time Frames")
    axs[i].set_ylabel("MFCC Coefficients")
    # axs[i].colorbar()
    
    fig.colorbar(im, ax=axs[i])  # Add color bar for reference
    
plt.show()
    
    