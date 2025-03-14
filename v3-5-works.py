import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
import os
import parquet_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AudioTranscriptionDataset(Dataset):
    def __init__(self, parquet_filename, alphabet, char_to_index): # Pass alphabet to constructor
        self.parquet_filename = parquet_filename
        self.data_list = []
        self.alphabet = alphabet # Store alphabet
        self.char_to_index = char_to_index # Use provided char_to_index mapping
        
        dataframe = parquet_dataframe.dataframe_from_parquet(parquet_filename)#.head(500) # Load a subset for testing
            
        for i in range(0, len(dataframe)):
            self.data_list.append({
                'audio': dataframe.iloc[i]['audio']['bytes'],
                'text': dataframe.iloc[i]['text'].lower(), # Basic processing - lowercase
                'words': dataframe.iloc[i]['words'],
                'word_start': dataframe.iloc[i]['word_start'],
                'word_end': dataframe.iloc[i]['word_end'],
            })
            if len([char for char in dataframe.iloc[i]['text'].lower() if char not in alphabet]) > 0:
                print("Warning: Unknown characters found in transcript:", dataframe.iloc[i]['text'].lower())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        audio_bytes = data_item['audio']
        transcript_text = data_item['text']
        
        audio_ndarray = np.array(parquet_dataframe.audio_bytes_to_ndarray(audio_bytes))

        # waveform, sample_rate = torchaudio.load(audio_filepath)
        waveform = torch.tensor(audio_ndarray).float() # Convert to tensor
        
        mfccs = mfcc_transform(waveform) # Apply MFCC transform

        # Convert transcript to numerical indices
        transcript_indices = [self.char_to_index[char] for char in transcript_text if char in self.char_to_index] # Handle unknown chars if needed
        transcript_tensor = torch.tensor(transcript_indices, dtype=torch.int32)
        
        # Store alignment info (for potential later use)
        words = data_item['words']
        word_start_times = data_item['word_start']
        word_end_times = data_item['word_end']

        alignment_info = { # Store alignment data in a dictionary
            'words': words,
            'word_start_times': word_start_times,
            'word_end_times': word_end_times
        }
        
        # print(f"--- Sample index: {idx} ---") # Indicate sample index
        # print("Waveform shape:", waveform.shape)
        # print("MFCCs shape:", mfccs.shape)
        # print("Transcript text:", transcript_text)
        # print("Transcript indices:", transcript_indices)
        # print("Transcript tensor:", transcript_tensor)

        return mfccs, transcript_tensor, alignment_info # Return alignment data as well


from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd_aligned(batch, blank_index): # Updated collate function
    mfccs_batch = [item[0].transpose(0, 1) for item in batch]
    # print("MFCCs batch shape:", mfccs_batch[0].shape)
    # mfccs_batch = [item[0] for item in batch]
    transcript_batch = [item[1] for item in batch]
    alignment_info_batch = [item[2] for item in batch] # Collect alignment info

    mfccs_batch_padded = pad_sequence(mfccs_batch, batch_first=True, padding_value=0.0)
    transcript_batch_padded = pad_sequence(transcript_batch, batch_first=True, padding_value=blank_index)

    input_lengths = torch.tensor([mfcc.size(0) for mfcc in mfccs_batch], dtype=torch.int32)
    target_lengths = torch.tensor([transcript.size(0) for transcript in transcript_batch], dtype=torch.int32)
    
    # print("--- Batch collated ---")
    # print("MFCCs batch padded shape:", mfccs_batch_padded.shape)
    # print("Transcript batch padded shape:", transcript_batch_padded.shape)
    # print("Input lengths:", input_lengths)
    # print("Target lengths:", target_lengths)

    return mfccs_batch_padded, transcript_batch_padded, input_lengths, target_lengths, alignment_info_batch # Return alignment info

import torch.nn as nn

class TransformerTranscriber(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=512, num_encoder_layers=4, nhead=8, dropout=0.1): # Increased hidden_dim, num_encoder_layers, nhead
        super(TransformerTranscriber, self).__init__()

        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_encoder_layers) # Increased layers
        self.output_fc = nn.Linear(hidden_dim, output_size)

    def forward(self, src):
        src = self.input_proj(src)
        encoder_output = self.transformer_encoder(src)
        output = self.output_fc(encoder_output)
        return output


sample_rate = 16000

n_mfcc = 40  # Number of MFCC coefficients to compute
melkwargs = {
    # 'n_mels': 50,  # Number of Mel filter banks
    # 'n_fft': 512,  # Number of FFT points
}
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs) # Initialize MFCC transform

# Define character set - example (expand this for real use)
alphabet = "abcdefghijklmnopqrstuvwxyz .',?!" # Space character
char_to_index = {char: index for index, char in enumerate(alphabet)}
index_to_char = {index: char for index, char in enumerate(alphabet)}
blank_index = len(alphabet) # Blank token index is last

output_size_ctc = len(alphabet) + 1 # +1 for blank

print("Alphabet:", alphabet)
print("Char to Index:", char_to_index)
print("Index to Char:", index_to_char)
print("Blank Index:", blank_index)
print("Output Size (CTC):", output_size_ctc)

# quit()

# Initialize Dataset
parquet_filename = "valid-00000-of-00001.parquet" # Replace
dataset_aligned = AudioTranscriptionDataset(parquet_filename, alphabet, char_to_index) # Pass alphabet

batch_size = 10 # Adjust as needed
train_dataloader_aligned = DataLoader(
    dataset_aligned,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn_padd_aligned(batch, blank_index) # Use updated collate function
)

####################

# Model Parameters - IMPORTANT: Update these when initializing your model
input_size_transformer = n_mfcc # Keep n_mfcc=40
hidden_dim_transformer = 512  # Increased hidden dimension
output_size_transformer = output_size_ctc
num_encoder_layers_transformer = 4 # Increased encoder layers
nhead_transformer = 8 # Increased attention heads

model_transformer = TransformerTranscriber(
    input_size=input_size_transformer,
    output_size=output_size_transformer,
    hidden_dim=hidden_dim_transformer,
    num_encoder_layers=num_encoder_layers_transformer,
    nhead=nhead_transformer
)

import torch.optim as optim
import torch.nn.functional as F

ctc_loss = nn.CTCLoss()

learning_rate = 0.0001

model = model_transformer

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adjust learning rate

# Training Epochs
num_epochs = 3 # Start with a small number for testing

history = {'train_loss': []}
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    total_loss = 0

    for batch_idx, (mfccs_batch, transcript_batch, input_lengths, target_lengths, _) in enumerate(train_dataloader_aligned):
        optimizer.zero_grad()

        transformer_output = model(mfccs_batch)
        log_probs = F.log_softmax(transformer_output, dim=2) # Model output, log_softmax
        log_probs = log_probs.transpose(0, 1) # [seq_len, batch_size, num_classes] - CTC loss input

        loss = ctc_loss(log_probs, transcript_batch, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        if batch_idx % 10 == 0: # Print progress every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader_aligned)}], Loss: {avg_loss:.4f}")
        
        history['train_loss'].append(avg_loss)
        
    avg_epoch_loss = total_loss / len(train_dataloader_aligned)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_epoch_loss:.4f}")
    # (Optional) Save model checkpoint here

print("Training finished!")

# Save plot of training loss
plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("v3_train_loss_plot.png")
# plt.show()

# Save model checkpoint
torch.save(model.state_dict(), "v3_model.pth")

# # Load model checkpoint
# model.load_state_dict(torch.load("v3_model.pth"))

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
model.eval() # Set to evaluation mode
with torch.no_grad():
    while True:
        sample_mfccs, transcript_tensor, _, _, _ = next(iter(train_dataloader_aligned)) # Get a sample batch (or load a single audio file)
        model_output = model(sample_mfccs) # [batch_size, seq_len, output_size]
        predicted_text = greedy_decode_ctc(model_output[0].unsqueeze(0), index_to_char, blank_index) # Decode first sample in batch
        print("Predicted Text (Greedy Decoding):", predicted_text)
        print("Actual Text:", "".join([index_to_char[index.item()] for index in transcript_tensor[0] if index != blank_index]))
        
        # Plot sample_mfccs 
        plt.figure(figsize=(12, 6))
        plt.imshow(sample_mfccs[0].T, origin='lower', aspect='auto', cmap='viridis')
        plt.xlabel("Time Frames")
        plt.ylabel("MFCC Coefficients")
        plt.title("MFCC Features")
        plt.colorbar()
        plt.show()
        plt.close()
        input("Press Enter to continue...")
        plt.clf()
        plt.cla()

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
        plt.close()
        input("Press Enter to continue...")
        plt.clf()
        plt.cla()