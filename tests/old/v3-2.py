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

class SimpleRNNTranscriber(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNNTranscriber, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True) # Bidirectional LSTM
        self.fc = nn.Linear(hidden_size * 2, output_size) # Output layer

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size] (MFCCs)
        out, _ = self.rnn(x) # out shape: [batch_size, seq_len, hidden_size * 2 (bidirectional)]
        out = self.fc(out)   # out shape: [batch_size, seq_len, output_size]
        return out

# Model parameters (you'll need to adjust these)
input_size = n_mfcc  # MFCC feature size
hidden_size = 128
output_size = 28  # Example: 26 letters + space + <blank> token (for CTC later)
model = SimpleRNNTranscriber(input_size, hidden_size, output_size)

# Prepare a dummy input (for a single sample)
dummy_input = dummy_mfccs[0].transpose(1, 2) # [1, time_frames, n_mfcc] - RNN expects [batch_size, seq_len, input_size]

# Get model output
output = model(dummy_input)
print("Model output shape:", output.shape) # [1, seq_len, output_size]