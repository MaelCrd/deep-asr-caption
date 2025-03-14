import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
import os
import parquet_dataframe
import pandas as pd
import numpy as np

class AudioTranscriptionDataset(Dataset):
    def __init__(self, parquet_filename, alphabet, char_to_index): # Pass alphabet to constructor
        self.parquet_filename = parquet_filename
        self.audios = []
        self.transcripts = []
        self.alphabet = alphabet # Store alphabet
        # self.char_to_index = {char: index for index, char in enumerate(alphabet)}
        self.char_to_index = char_to_index # Use provided char_to_index mapping
        
        dataframe = parquet_dataframe.dataframe_from_parquet(parquet_filename).head(100) # Load a subset for testing
            
        for i in range(0, len(dataframe)):
            self.audios.append(dataframe.iloc[i]['audio']['bytes'])
            self.transcripts.append(dataframe.iloc[i]['text'].lower()) # Basic processing - lowercase

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio_bytes = self.audios[idx]
        transcript_text = self.transcripts[idx]
        
        audio_ndarray = np.array(parquet_dataframe.audio_bytes_to_ndarray(audio_bytes))

        # waveform, sample_rate = torchaudio.load(audio_filepath)
        waveform = torch.tensor(audio_ndarray).float() # Convert to tensor
        
        mfccs = mfcc_transform(waveform) # Apply MFCC transform

        # Convert transcript to numerical indices
        transcript_indices = [self.char_to_index[char] for char in transcript_text if char in self.char_to_index] # Handle unknown chars if needed
        transcript_tensor = torch.tensor(transcript_indices, dtype=torch.int32)

        return mfccs, transcript_tensor


from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd(batch, blank_index):
    # mfccs_batch = [item[0].transpose(1, 2) for item in batch] # Transpose MFCCs to [seq_len, n_mfcc]
    mfccs_batch = [item[0].transpose(0, 1) for item in batch] # Changed but not sure
    transcript_batch = [item[1] for item in batch]

    mfccs_batch_padded = pad_sequence(mfccs_batch, batch_first=True, padding_value=0.0) # Pad MFCCs with 0s
    transcript_batch_padded = pad_sequence(transcript_batch, batch_first=True, padding_value=blank_index) # Pad transcripts with blank index

    input_lengths = torch.tensor([mfcc.size(0) for mfcc in mfccs_batch], dtype=torch.int32) # Original lengths before padding
    target_lengths = torch.tensor([transcript.size(0) for transcript in transcript_batch], dtype=torch.int32)

    return mfccs_batch_padded, transcript_batch_padded, input_lengths, target_lengths

import torch.nn as nn

class CTC_RNNTranscriber(nn.Module): # Renamed model
    def __init__(self, input_size, hidden_size, output_size): # output_size now includes blank token
        super(CTC_RNNTranscriber, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # Output layer still linear

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        # No softmax here! CTC loss expects raw logits
        return out


sample_rate = 16000

n_mfcc = 10  # Number of MFCC coefficients to compute
melkwargs = {
    'n_mels': 85,
}
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs) # Initialize MFCC transform

# Define character set - example (expand this for real use)
alphabet = "abcdefghijklmnopqrstuvwxyz " # Space character
char_to_index = {char: index for index, char in enumerate(alphabet)}
index_to_char = {index: char for index, char in enumerate(alphabet)}
blank_index = len(alphabet) # Blank token index is last

output_size_ctc = len(alphabet) + 1 # +1 for blank

# Initialize Dataset
parquet_filename = "valid-00000-of-00001.parquet" # Replace
dataset = AudioTranscriptionDataset(parquet_filename, alphabet, char_to_index) # Pass alphabet

batch_size = 5 # Adjust as needed
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn_padd(batch, blank_index)) # Use a custom collate function for padding
# You'll need to implement collate_fn_padd to handle variable length sequences within batches (MFCCs and transcripts) - see next step

####################

# Model parameters (you'll need to adjust these)
input_size = n_mfcc  # MFCC feature size
hidden_size = 128
learning_rate = 0.001
model_ctc = CTC_RNNTranscriber(input_size, hidden_size, output_size_ctc)

import torch.optim as optim
import torch.nn.functional as F

ctc_loss = nn.CTCLoss()

# Optimizer
optimizer = optim.Adam(model_ctc.parameters(), lr=learning_rate) # Adjust learning rate

# Training Epochs
num_epochs = 10 # Start with a small number for testing

history = {'train_loss': []}
for epoch in range(num_epochs):
    model_ctc.train() # Set model to training mode
    total_loss = 0

    for batch_idx, (mfccs_batch, transcript_batch, input_lengths, target_lengths) in enumerate(train_dataloader):
        optimizer.zero_grad()

        log_probs = F.log_softmax(model_ctc(mfccs_batch), dim=2) # Model output, log_softmax
        log_probs = log_probs.transpose(0, 1) # [seq_len, batch_size, num_classes] - CTC loss input

        loss = ctc_loss(log_probs, transcript_batch, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0: # Print progress every 10 batches
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")

    avg_epoch_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}")
    # (Optional) Save model checkpoint here
    history['train_loss'].append(avg_epoch_loss)

print("Training finished!")

# Save plot of training loss
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("v3_train_loss_plot.png")
# plt.show()

# Save model checkpoint
torch.save(model_ctc.state_dict(), "model_ctc.pth")

def greedy_decode_ctc(model_output, index_to_char, blank_index):
    """Greedy decoding for CTC output."""
    predicted_indices = torch.argmax(model_output, dim=2).squeeze(0).cpu().numpy() # Get indices with max prob for each time step
    predicted_chars = []
    for index in predicted_indices:
        if index != blank_index: # Skip blank tokens
            if not predicted_chars or index != predicted_chars[-1]: # Remove consecutive repeats
                predicted_chars.append(index_to_char[index])
    return "".join(predicted_chars)

# Example Usage (after training, in evaluation or inference)
model_ctc.eval() # Set to evaluation mode
with torch.no_grad():
    sample_mfccs, _, _, _ = next(iter(train_dataloader)) # Get a sample batch (or load a single audio file)
    model_output = model_ctc(sample_mfccs) # [batch_size, seq_len, output_size]
    predicted_text = greedy_decode_ctc(model_output[0].unsqueeze(0), index_to_char, blank_index) # Decode first sample in batch
    print("Predicted Text (Greedy Decoding):", predicted_text)

    # (After model forward pass)
    probabilities = F.softmax(model_output[0], dim=1).cpu().numpy() # Softmax to get probabilities, first sample in batch
    plt.figure(figsize=(12, 6))
    plt.imshow(probabilities.T, origin='lower', aspect='auto', cmap='viridis') # Spectrogram-like view
    plt.yticks(range(output_size_ctc), list(alphabet) + ["<blank>"]) # Y-axis labels
    plt.xlabel("Time Frames")
    plt.ylabel("Characters")
    plt.title("CTC Output Probabilities (Basic Frame-Level 'Alignment')")
    plt.colorbar()
    plt.show()