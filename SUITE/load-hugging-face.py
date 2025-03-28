import datasets
import datasets.hub
from torch.utils.data import DataLoader

# Authenticate to the Hugging Face Datasets library
# from huggingface_hub import login
# login()

cv_17 = datasets.load_dataset(
    "mozilla-foundation/common_voice_17_0", 
    "en",
    split=datasets.Split.TRAIN, 
    streaming=True,
    # trust_remote_code=True, 
)

print(cv_17.take(100000))

print(cv_17.get(0))

# dataloader = DataLoader(cv_17, batch_size=32)

quit()

iterator = iter(cv_17)

print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))

for i in range(5):
    row = next(iterator)

    # Save audio to file
    import os
    import torchaudio
    import torch

    # Save the audio to a file
    audio_array = torch.tensor(row['audio']['array'])
    sample_rate = row['audio']['sampling_rate']
    
    print(row['sentence'])

    # Make the audio stereo
    audio_array = torch.stack([audio_array, audio_array])
    torchaudio.save(f"audio({i})_cm.mp3", audio_array, sample_rate)