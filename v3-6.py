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
        
        dataframe = parquet_dataframe.dataframe_from_parquet(parquet_filename) # Load a subset for testing
            
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

# Add <sos> and <eos> tokens to your alphabet and mappings
alphabet_attention = "abcdefghijklmnopqrstuvwxyz .',?!" # Add <sos> and <eos>
char_to_index_attention = {char: index for index, char in enumerate(alphabet_attention)}
index_to_char_attention = {index: char for index, char in enumerate(alphabet_attention)}
# Add <sos> and <eos> tokens
char_to_index_attention['<sos>'] = len(char_to_index_attention)
index_to_char_attention[len(index_to_char_attention)] = '<sos>'
char_to_index_attention['<eos>'] = len(char_to_index_attention)
index_to_char_attention[len(index_to_char_attention)] = '<eos>'
blank_index_attention = -1 # No blank token needed for attention, set to -1 or None
sos_index = char_to_index_attention['<sos>']
eos_index = char_to_index_attention['<eos>']
output_size_attention = len(alphabet_attention) + 2 # Add 2 for <sos> and <eos> tokens

# # Print to verify
# print("Alphabet (Attention):", alphabet_attention)
# print("Char to index (Attention):", char_to_index_attention)
# print("Index to char (Attention):", index_to_char_attention)
# print("Output size (Attention):", output_size_attention)

# print("Attention Alphabet:", alphabet_attention)
# print("Attention Char to Index:", char_to_index_attention)
# print("Attention Index to Char:", index_to_char_attention)
# print("SOS Index:", sos_index)
# print("EOS Index:", eos_index)
# print("Attention Output Size:", output_size_attention)


def collate_fn_padd_attention(batch): # No blank_index needed here
    mfccs_batch = [item[0].transpose(0, 1) for item in batch] # You might need to adjust transpose if needed
    transcript_batch = [item[1] for item in batch] # Original transcript indices

    # Create decoder inputs and outputs
    trg_input_batch = []
    trg_output_batch = []
    for transcript in transcript_batch:
        trg_input = torch.cat([torch.tensor([sos_index], dtype=torch.int32), transcript], dim=0) # Prepend <sos>
        trg_output = torch.cat([transcript, torch.tensor([eos_index], dtype=torch.int32)], dim=0) # Append <eos>
        trg_input_batch.append(trg_input)
        trg_output_batch.append(trg_output)


    mfccs_batch_padded = pad_sequence(mfccs_batch, batch_first=True, padding_value=0.0)
    trg_input_batch_padded = pad_sequence(trg_input_batch, batch_first=True, padding_value=char_to_index_attention[' ']) # Pad decoder input with space index (or any padding index that makes sense for your alphabet)
    trg_output_batch_padded = pad_sequence(trg_output_batch, batch_first=True, padding_value=char_to_index_attention[' ']) # Pad decoder output with space index

    input_lengths = torch.tensor([mfcc.size(0) for mfcc in mfccs_batch], dtype=torch.int32)
    trg_input_lengths = torch.tensor([trg_input.size(0) for trg_input in trg_input_batch], dtype=torch.int32) # Lengths of decoder inputs
    trg_output_lengths = torch.tensor([trg_output.size(0) for trg_output in trg_output_batch], dtype=torch.int32) # Lengths of decoder outputs

    # print("--- Batch collated (Attention) ---")
    # print("MFCCs batch padded shape:", mfccs_batch_padded.shape)
    # print("Decoder input batch padded shape:", trg_input_batch_padded.shape)
    # print("Decoder output batch padded shape:", trg_output_batch_padded.shape)
    # print("Input lengths:", input_lengths)
    # print("Decoder Input lengths:", trg_input_lengths)
    # print("Decoder Output lengths:", trg_output_lengths)
    
    # print("SOS Index:", sos_index) # ADD THIS PRINT
    # print("EOS Index:", eos_index) # ADD THIS PRINT

    # print("Decoder Input Batch Padded (example sample):\n", trg_input_batch_padded[0]) # ADD THIS PRINT - show numerical values for the first sample in the batch
    # print("Decoder Output Batch Padded (example sample):\n", trg_output_batch_padded[0]) # ADD THIS PRINT - show numerical values for the first sample in the batch

    return mfccs_batch_padded, trg_input_batch_padded, trg_output_batch_padded, input_lengths, trg_output_lengths # Return decoder input and output

import torch.nn as nn

class AttentionTranscriber(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=512, num_encoder_layers=2, nhead=8, dropout=0.1, num_decoder_layers=1, attention_mechanism='dot'): # Added decoder layers, attention_mechanism
        super(AttentionTranscriber, self).__init__()

        # Encoder (Transformer Encoder - can reuse your existing encoder)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_encoder_layers)

        # Decoder (Simple RNN Decoder - LSTM for example)
        self.decoder_rnn = nn.LSTM(output_size + hidden_dim, hidden_dim, num_decoder_layers, batch_first=True) # Input: embedding + context vector, Output: hidden state
        self.output_fc = nn.Linear(hidden_dim, output_size) # Output layer to character probabilities

        # Attention Mechanism (Dot Product Attention - you can try others like Additive Attention)
        self.attention_mechanism = attention_mechanism
        if attention_mechanism == 'dot':
            self.attention = DotProductAttention() # Define DotProductAttention class (see below)
        elif attention_mechanism == 'additive':
            self.attention = AdditiveAttention(hidden_dim) # Define AdditiveAttention class (see below)
        else:
            raise ValueError(f"Attention mechanism '{attention_mechanism}' not supported.")

        self.embedding = nn.Embedding(output_size, output_size) # Character embedding - for decoder input

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # src: [batch_size, src_seq_len, input_size] (MFCCs)
        # trg: [batch_size, trg_seq_len] (Target text indices) - shifted right input to decoder (start with <sos> token)

        # Encoder
        encoder_output = self.encode(src, src_mask) # Get encoder outputs - [batch_size, src_seq_len, hidden_dim]

        # Initialize decoder hidden state (e.g., from encoder final state or zeros - simplified here to zeros)
        decoder_hidden = None # Initialize to None for LSTM to start with zeros

        # Decoder - step-by-step generation
        batch_size, trg_seq_len = trg.size()
        decoder_outputs = []
        context_vector = torch.zeros(batch_size, self.decoder_rnn.hidden_size).to(src.device) # Initialize context vector

        for t in range(trg_seq_len):
            decoder_input = trg[:, t] # Get current target token index
            decoder_input_embedded = self.embedding(decoder_input).unsqueeze(1) # [batch_size, 1, output_size] - Embed current token
            decoder_input_combined = torch.cat([decoder_input_embedded, context_vector.unsqueeze(1)], dim=2) # Concatenate embedding and context [batch_size, 1, output_size + hidden_dim]

            decoder_output_rnn, decoder_hidden = self.decoder_rnn(decoder_input_combined, decoder_hidden) # RNN step
            # decoder_output_rnn shape: [batch_size, 1, hidden_dim]

            # Attention Calculation
            context_vector, attention_weights = self.attention(decoder_output_rnn, encoder_output) # Calculate context vector using attention
            # context_vector shape: [batch_size, hidden_dim], attention_weights: [batch_size, src_seq_len]

            output = self.output_fc(decoder_output_rnn) # Linear layer to get logits
            # output shape: [batch_size, 1, output_size]

            decoder_outputs.append(output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [batch_size, trg_seq_len, output_size]
        return decoder_outputs # Return decoder outputs (logits)

    def encode(self, src, src_mask):
        src = self.input_proj(src) # Project input features to hidden dimension
        encoder_output = self.transformer_encoder(src, mask=src_mask) # Transformer encoder
        return encoder_output


# --- Attention Mechanism Classes --- (Simple Dot Product Attention)
class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, values):
        """
        Dot product attention.
        query: [batch_size, 1, hidden_dim] - Decoder hidden state (at current step)
        values: [batch_size, src_seq_len, hidden_dim] - Encoder outputs (all time steps)
        """
        attention_weights = torch.bmm(query, values.transpose(1, 2)).squeeze(1) # [batch_size, src_seq_len] - Dot product, remove dim=1
        attention_probs = F.softmax(attention_weights, dim=1) # [batch_size, src_seq_len] - Softmax over source sequence

        context_vector = torch.bmm(attention_probs.unsqueeze(1), values).squeeze(1) # [batch_size, hidden_dim] - Weighted sum of values (encoder outputs)
        return context_vector, attention_probs

##################""

sample_rate = 16000

n_mfcc = 40  # Number of MFCC coefficients to compute
melkwargs = {
    'n_mels': 50,  # Number of Mel filter banks
    # 'n_fft': 512,  # Number of FFT points
}
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs) # Initialize MFCC transform

output_size_ctc = len(alphabet_attention) # Alphabet size with <sos> and <eos> tokens (no blank token)

# quit()

# Initialize Dataset
parquet_filename = "valid-00000-of-00001.parquet" # Replace
dataset_aligned = AudioTranscriptionDataset(parquet_filename, alphabet_attention, char_to_index_attention) # Pass alphabet

batch_size = 10 # Adjust as needed
# Update DataLoader to use the new collate function and alphabet/output_size
train_dataloader_attention = DataLoader(
    dataset_aligned, # Assuming you are still using dataset_aligned
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_padd_attention
)

######################

# --- Example Usage and Initialization ---
input_size_attention = n_mfcc # Keep n_mfcc=40
hidden_dim_attention = 512
num_encoder_layers_attention = 2
nhead_attention = 8
num_decoder_layers_attention = 1 # Start with a single decoder layer
attention_mechanism_type = 'dot' # Or 'additive' if you implement AdditiveAttention

model_attention = AttentionTranscriber(
    input_size=input_size_attention,
    output_size=output_size_attention,
    hidden_dim=hidden_dim_attention,
    num_encoder_layers=num_encoder_layers_attention,
    nhead=nhead_attention,
    num_decoder_layers=num_decoder_layers_attention,
    attention_mechanism=attention_mechanism_type
)

# Load checkpoint if needed
model_attention.load_state_dict(torch.load("v3_model_attention_checkpoint.pth"))

import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.005

# Optimizer
optimizer_attention = optim.Adam(model_attention.parameters(), lr=learning_rate) # Adjust learning rate

# Loss Function - CrossEntropyLoss (for sequence generation)
criterion_attention = nn.CrossEntropyLoss(ignore_index=char_to_index_attention[' ']) # Ignore padding index in loss

# Training Epochs
num_epochs = 3 # Start with a small number for testing

TRAIN = False
if TRAIN:
    history = {'train_loss': []}
    for epoch in range(num_epochs):
        model_attention.train()
        total_loss = 0

        for batch_idx, (mfccs_batch, trg_input_batch, trg_output_batch, input_lengths, trg_output_lengths) in enumerate(train_dataloader_attention):
            optimizer_attention.zero_grad()

            # Forward pass - Attention model expects src, trg_input
            decoder_output = model_attention(mfccs_batch, trg_input_batch) # [batch_size, trg_seq_len, output_size]

            # Reshape for CrossEntropyLoss - loss expects (N, C, L) input and (N, L) target
            decoder_output_reshape = decoder_output.view(-1, decoder_output.size(-1)) # [batch_size * trg_seq_len, output_size]
            trg_output_reshape = trg_output_batch.view(-1).long() # [batch_size * trg_seq_len]

            loss_attention = criterion_attention(decoder_output_reshape, trg_output_reshape)
            loss_attention.backward()
            
            torch.nn.utils.clip_grad_norm_(model_attention.parameters(), max_norm=1) # Or try max_norm=5 or 10 if 1 is too restrictive
            optimizer_attention.step()

            total_loss += loss_attention.item()

            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader_attention)}], Loss (Attention): {avg_loss:.4f}")
                torch.save(model_attention.state_dict(), "v3_model_attention_checkpoint.pth")
                
        avg_epoch_loss = total_loss / len(train_dataloader_attention)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss (Attention): {avg_epoch_loss:.4f}")
        # (Optional) Save attention model checkpoint
        torch.save(model_attention.state_dict(), "v3_model_attention_checkpoint.pth")

    print("Attention-based Training finished!")
    # Save model checkpoint
    torch.save(model_attention.state_dict(), "v3_model_attention.pth")

    # # Save plot of training loss
    # plt.figure()
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.title("Training Loss")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig("v3_train_loss_plot.png")
    # # plt.show()
else:
    print("Training skipped.")
    print("Loading model_attention from checkpoint...")
    model_attention.load_state_dict(torch.load("v3_model_attention_checkpoint.pth"))


# # Load model checkpoint
# model_attention.load_state_dict(torch.load("v3_model_attention.pth"))

def greedy_decode_attention(model, src, max_len, sos_index, eos_index, index_to_char):
    """Greedy decoding for attention-based model."""
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src, src_mask=None) # Encode input audio

        batch_size = src.size(0)
        decoder_input = torch.tensor([sos_index] * batch_size, dtype=torch.int32).unsqueeze(1).to(src.device) # Start with <sos> tokens [batch_size, 1]
        decoder_hidden = None # Initialize decoder hidden state
        predicted_indices = []

        context_vector = torch.zeros(batch_size, model.decoder_rnn.hidden_size).to(src.device) # Initialize context vector

        for _ in range(max_len): # Generate up to max_len tokens
            decoder_input_embedded = model.embedding(decoder_input) # [batch_size, 1, output_size]
            # print("decoder_input_embedded", decoder_input_embedded.shape)
            # decoder_input_embedded = decoder_input_embedded.unsqueeze(1)
            # print("decoder_input_embedded", decoder_input_embedded.shape)
            # print("context_vector", context_vector.shape)
            # print("context_vector.unsqueeze(1)", context_vector.unsqueeze(1).shape)
            decoder_input_combined = torch.cat([decoder_input_embedded, context_vector.unsqueeze(1)], dim=2) # [batch_size, 1, output_size + hidden_dim]

            decoder_output_rnn, decoder_hidden = model.decoder_rnn(decoder_input_combined, decoder_hidden)
            context_vector, attention_weights = model.attention(decoder_output_rnn, encoder_output)
            output = model.output_fc(decoder_output_rnn) # [batch_size, 1, output_size]
            predicted_token_index = torch.argmax(output, dim=2) # Greedy choice - [batch_size, 1]

            predicted_indices.append(predicted_token_index.cpu().numpy())
            decoder_input = predicted_token_index # Use predicted token as next input

            if all(token == eos_index for token in predicted_token_index.flatten()): # Stop if all beams predict <eos>
                break


        predicted_indices = np.concatenate(predicted_indices, axis=1) # [batch_size, seq_len]
        predicted_texts = []
        for sample_indices in predicted_indices:
            text = ""
            for index in sample_indices:
                if index == eos_index:
                    break # Stop at <eos>
                text += index_to_char[index]
            predicted_texts.append(text)

        return predicted_texts

# Example Usage - after attention model training
model_attention.eval()
with torch.no_grad():
    while True:
        sample_mfccs_attention, _, _, _, _ = next(iter(train_dataloader_attention)) # Get a sample batch
        max_decode_len = 200 # Set a maximum decoding length
        predicted_texts_attention = greedy_decode_attention(
            model_attention,
            sample_mfccs_attention,
            max_decode_len,
            sos_index,
            eos_index,
            index_to_char_attention
        )
        print("Predicted Texts (Attention, Greedy Decoding):", predicted_texts_attention)
        input("Press Enter to continue...")
