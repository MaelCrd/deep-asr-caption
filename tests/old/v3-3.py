import torch
from torchaudio.transforms import MFCC
import matplotlib.pyplot as plt

n_mfcc = 40  # Number of MFCC coefficients to compute
sample_rate = 16000

# Very simplified example - replace with real data later
dummy_mfccs = [torch.randn(1, n_mfcc, 100), torch.randn(1, n_mfcc, 150), torch.randn(1, n_mfcc, 80)] # List of MFCC tensors
dummy_transcripts = ["hello world", "pytorch is fun", "audio transcription"] # Corresponding text transcripts (not aligned)

# We'll need to tokenize text later, for now keep it as strings

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

# Model parameters (you'll need to adjust these)
input_size = n_mfcc  # MFCC feature size
hidden_size = 128

# Define character set - example (expand this for real use)
alphabet = "abcdefghijklmnopqrstuvwxyz " # Space character
char_to_index = {char: index for index, char in enumerate(alphabet)}
index_to_char = {index: char for index, char in enumerate(alphabet)}
blank_index = len(alphabet) # Blank token index is last

output_size_ctc = len(alphabet) + 1 # +1 for blank
model_ctc = CTC_RNNTranscriber(input_size, hidden_size, output_size_ctc)

# Prepare a dummy input (for a single sample)
dummy_input = dummy_mfccs[0].transpose(1, 2) # [1, time_frames, n_mfcc] - RNN expects [batch_size, seq_len, input_size]

# Get model output
output = model_ctc(dummy_input)
print("Model output shape:", output.shape) # [1, seq_len, output_size]


import torch.nn.functional as F

ctc_loss = nn.CTCLoss()

# Dummy data (replace with proper data loading later)
log_probs = F.log_softmax(model_ctc(dummy_input), dim=2) # Apply log_softmax along class dimension
target_transcript = dummy_transcripts[0] # "hello world"
target_indices = [char_to_index[char] for char in target_transcript] # Convert to indices
target_tensor = torch.tensor(target_indices, dtype=torch.int32)

input_lengths = torch.tensor([log_probs.size(1)], dtype=torch.int32) # Sequence length of input
target_lengths = torch.tensor([len(target_tensor)], dtype=torch.int32) # Length of target

# Need to expand dimensions for CTC loss to work with batch size of 1
log_probs = log_probs.transpose(0,1) # [seq_len, batch_size, num_classes] for CTCLoss input
target_tensor = target_tensor.unsqueeze(0) # [1, target_len] - batch dimension for targets
input_lengths = input_lengths.unsqueeze(0) # [1] - batch dimension for input lengths
target_lengths = target_lengths.unsqueeze(0) # [1] - batch dimension for target lengths


loss = ctc_loss(log_probs, target_tensor, input_lengths, target_lengths)
print("CTC Loss:", loss)