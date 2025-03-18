import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MFCC
from torchaudio.functional import resample
import os
import jiwer  # For WER calculation
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Import functional
from datetime import datetime

import librosa

import parquet_dataframe

# 3. Data Preprocessing
# 3.1 Audio Feature Extraction
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

# 3.2 Text Processing (Character Vocabulary and Tokenization)
characters = ["<blank>", " ", "'", ".", ",", "!", "?", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}

def tokenize_text(text):
    text = text.lower()
    tokens = [char_to_index[char] for char in text if char in char_to_index]
    return tokens

# 3.3 Custom Dataset and Data Loaders
class ASRDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrogram, transcript = self.dataset[idx]
        tokens = self.tokenizer(transcript)
        return spectrogram, torch.tensor(tokens)

def collate_fn_asr(batch):
    mfccs_batch = [item[0] for item in batch] # These are already transposed
    tokens_batch = [item[1] for item in batch]
    mfccs_untransposed_batch = [item[2] for item in batch] # Get untransposed MFCCs

    mfccs_padded = pad_sequence(mfccs_batch, batch_first=True)
    tokens_padded = pad_sequence(tokens_batch, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([mfcc.shape[1] for mfcc in mfccs_untransposed_batch]) # Use untransposed for length (shape[1] is time_frames)
    target_lengths = torch.tensor([len(tokens) for tokens in tokens_batch])

    return mfccs_padded, tokens_padded, input_lengths, target_lengths


files_source = []
# files_source.append("train-00000-of-00064.parquet")
# files_source.append("train-00001-of-00064.parquet")
# files_source.append("train-00002-of-00064.parquet")
# files_source.append("train-00003-of-00064.parquet")
# files_source.append("train-00004-of-00064.parquet")
# files_source.append("train-00005-of-00064.parquet")
# files_source.append("train-00006-of-00064.parquet")
# files_source.append("train-00007-of-00064.parquet")
# files_source.append("train-00008-of-00064.parquet")
# files_source.append("train-00009-of-00064.parquet")
# files_source.append("train-00010-of-00064.parquet")
# files_source.append("train-00011-of-00064.parquet")
# files_source.append("train-00012-of-00064.parquet")
# files_source.append("train-00013-of-00064.parquet")
# files_source.append("train-00014-of-00064.parquet")
# files_source.append("train-00015-of-00064.parquet")
# files_source.append("train-00016-of-00064.parquet")
# files_source.append("train-00063-of-00064.parquet")

files_source.append("valid-00000-of-00001.parquet")

dataset = []
for filename in files_source:
    filename = "dataset/" + filename
    print("Processing file:", filename)
    dataframe = parquet_dataframe.dataframe_from_parquet(filename) # Load a subset for testing

    print("Dataframe shape:", dataframe.shape)

    for i in range(0, len(dataframe)):
        spectrogram = custom_spectrogram(np.array(parquet_dataframe.audio_bytes_to_ndarray(dataframe.iloc[i]['audio']['bytes'])))
        dataset.append([
            spectrogram, # spectrogram
            dataframe.iloc[i]['text'], # transcript
        ])
    
    del dataframe

print("Total dataset size:", len(dataset))

print("Shuffling dataset...")
np.random.shuffle(dataset)

# Divide dataset into train and dev sets
split_ratio = 0.9
train_size = int(split_ratio * len(dataset))
dev_size = len(dataset) - train_size
train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])

train_asr_dataset = ASRDataset(train_dataset, mfcc_transform, tokenize_text)
dev_asr_dataset = ASRDataset(dev_dataset, mfcc_transform, tokenize_text)

batch_size = 32
train_dataloader = DataLoader(train_asr_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_asr)
dev_dataloader = DataLoader(dev_asr_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_asr)

# 8. Evaluation (Word Error Rate - WER)
def decode_predictions(log_probs, input_lengths):
    predicted_tokens = torch.argmax(log_probs, dim=2)
    decoded_sentences = []
    for i in range(predicted_tokens.shape[0]):
        tokens = predicted_tokens[i, :input_lengths[i]]
        sentence = ""
        for t in tokens:
            char = index_to_char[t.item()]
            if char != '<blank>': # Assuming no blank token in this simple example, adjust if needed
                sentence += char
        decoded_sentences.append(sentence)
    return decoded_sentences

from itertools import groupby

def ctc_decoder(log_probs, input_lengths):
    print(log_probs.shape)
    predicted_tokens = torch.argmax(log_probs, dim=2).cpu().numpy()
    
    decoded_sentences = []
    for i in range(predicted_tokens.shape[0]):
        tokens = predicted_tokens[i, :input_lengths[i]]
        # use groupby to find continuous same indexes
        tokens = [k for k, g in groupby(tokens)]
        sentence = ""
        for t in tokens:
            char = index_to_char[t.item()]
            if char != '<blank>': # Assuming no blank token in this simple example, adjust if needed
                sentence += char
        decoded_sentences.append(sentence)
    return decoded_sentences

def ctc_decoder_multiple(log_probs, input_lengths, n=3):
    # Get top n predictions
    # print(log_probs.shape)
    _, top_n_indices = torch.topk(log_probs, n, dim=2)
    top_n_indices = top_n_indices.cpu().numpy()
    
    # print(top_n_indices.shape)
    
    decoded_sentences = []
    for i in range(top_n_indices.shape[0]):
        tokens = top_n_indices[i, :input_lengths[i]]
        # print(tokens)
        decoded_sentences.append([])
        for j in range(n):
            tokens_j = tokens[:, j]
            # use groupby to find continuous same indexes
            tokens_j = [k for k, g in groupby(tokens_j)]
            sentence = ""
            for t in tokens_j:
                char = index_to_char[t.item()]
                if char != '<blank>': # Assuming no blank token in this simple example, adjust if needed
                    sentence += char
            decoded_sentences[-1].append(sentence)
    return decoded_sentences

def calculate_wer(predicted_sentences, reference_sentences):
    # wer = jiwer.wer(reference_sentences, predicted_sentences)
    # return wer

    # Ensure reference sentences are non-empty and properly tokenized
    reference_sentences = [sentence for sentence in reference_sentences if sentence.strip()]
    predicted_sentences = [sentence for sentence in predicted_sentences if sentence.strip()]
    
    if not reference_sentences or not predicted_sentences:
        return 1.0  # Return maximum WER if any list is empty

    if len(reference_sentences) != len(predicted_sentences):
        print(f"Length mismatch: {len(reference_sentences)} reference sentences, {len(predicted_sentences)} predicted sentences")
        # return 1.0  # Return maximum WER if lengths do not match
        return None

    wer = jiwer.wer(reference_sentences, predicted_sentences)
    return wer

def plot_history(history):
    plt.cla()
    plt.clf()
    plt.close()
    save_path = "asr_loss"
    if type(history) is dict:
        for key in history:
            plt.plot(history[key], label=key)
        plt.legend()
        plt.title("ASR Training Loss (Train vs. Val)")
        save_path += "_train_val.png"
    else:
        plt.plot(history)
        plt.title("ASR Training Loss (batch)")
        save_path += "_batch.png"
    plt.ylabel("Loss")
    plt.savefig(save_path)
    # plt.show()

# 4. Model Definition (Simple RNN-based ASR)
class MediumASR(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(MediumASR, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # RNN Layers
        self.lstm1 = nn.LSTM(160, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm4 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm5 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

        # Dense Layers
        self.dense1 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, output_dim + 1)  # +1 for blank

    def forward(self, x):
        # Expand dimensions
        x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # print("x.shape", x.shape)
        # print("x.shape before permute:", x.shape)  # torch.Size([32, 32, 842, 5])

        # Reshape for RNN
        x = x.permute(0, 2, 1, 3)  # Swap feature and channel dimensions
        # print("x.shape after permute:", x.shape)  # torch.Size([32, 842, 32, 5])
        batch_size, time_steps, channels, features = x.size()
        # print(f"{batch_size=}, {time_steps=}, {channels=}, {features=}")
        # Calculate the correct input size for the LSTM
        lstm_input_size = channels * features  # 32 * 5 = 160
        # print(f"{lstm_input_size=}")

        # x = x.reshape(batch_size, time_steps, channels * features)  # Combine channels and features # 160
        x = x.reshape(batch_size, time_steps, lstm_input_size)

        # print("Reshaped x shape:", x.shape)

        # RNN Layers
        x, _ = self.lstm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.lstm2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.lstm3(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.lstm4(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.lstm5(x)


        # Dense Layers
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification Layer
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)  # Using log_softmax for numerical stability with CTCLoss
        return x

input_dim = n_mfcc
# hidden_dim = 512 # big : 810
output_dim = len(characters)
model = MediumASR(input_dim, output_dim)
print(model)

# # Load model checkpoint if needed
# model.load_state_dict(torch.load("simple_asr_model_checkpoint.pth"))
# print(">>> Model loaded from checkpoint.")

# 5. Loss Function and Optimizer (CTC Loss and Adam)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 6. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
criterion.to(device)

TRAIN = False # Set to True to train the model

def main():
    if TRAIN:
        history_batches = []
        history_epochs = {"train": [], "val": []}
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            batch_group_loss = 0.0
            for batch_idx, (mfccs_padded, tokens_padded, input_lengths, target_lengths) in enumerate(train_dataloader):
                mfccs_padded = mfccs_padded.to(device)
                tokens_padded = tokens_padded.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                optimizer.zero_grad()
                # log_probs = model(mfccs_padded, input_lengths)
                log_probs = model(mfccs_padded)
                log_probs = log_probs.transpose(0, 1) # Time x Batch x Vocab
                # Inside your training loop, before the CTCLoss call:
                # print("log_probs shape:", log_probs.shape)
                # print("input_lengths:", input_lengths)
                # print("target_lengths:", target_lengths)
                # Correct input_lengths here:
                input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, tokens_padded, input_lengths, target_lengths)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients to a maximum norm of 1.0 (adjust max_norm if needed)
                optimizer.step()
                
                train_loss += loss.item()
                batch_group_loss += loss.item()

                batch_interval = 200
                if batch_idx % batch_interval == 0 and batch_idx > 0:
                    # avg_loss = train_loss / (batch_idx + 1)
                    avg_loss = batch_group_loss / batch_interval # Print average loss over last few batches
                    batch_group_loss = 0.0
                    print(f"[{datetime.now().strftime("%d-%m %H:%M")}] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Train Loss (last batch group): {avg_loss:.4f}")
                    # torch.save(model.state_dict(), "asr_v3.pth")
                    history_batches.append(avg_loss)
                    plot_history(history_batches)

            avg_epoch_loss = train_loss / len(train_dataloader)
            print(f"[{datetime.now().strftime("%d-%m %H:%M")}] Epoch [{epoch+1}/{num_epochs}] - Average Train Loss: {avg_epoch_loss:.4f}")
            history_epochs["train"].append(avg_epoch_loss)

            # 7. Evaluation during training (Validation Loss)
            model.eval()
            dev_loss = 0.0
            with torch.no_grad():
                for batch_idx, (mfccs_padded, tokens_padded, input_lengths, target_lengths) in enumerate(dev_dataloader):
                    mfccs_padded = mfccs_padded.to(device)
                    tokens_padded = tokens_padded.to(device)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)

                    # log_probs = model(mfccs_padded, input_lengths)
                    log_probs = model(mfccs_padded)
                    log_probs = log_probs.transpose(0, 1)
                    # Correct input_lengths here:
                    input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long, device=device)
                    loss = criterion(log_probs, tokens_padded, input_lengths, target_lengths)
                    dev_loss += loss.item()

            avg_dev_loss = dev_loss / len(dev_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_dev_loss:.4f}")
            history_epochs["val"].append(avg_dev_loss)
            
            plot_history(history_epochs)
            
            # Save model checkpoint
            torch.save(model.state_dict(), "asr_v3.pth")

        print("Training finished!")

        # Save model checkpoint
        torch.save(model.state_dict(), "asr_v3.pth")
    else :
        model.load_state_dict(torch.load("asr_v3.pth"))
        model.eval()


def evaluate():
    model.eval()
    total_wer = 0.0
    num_utterances = 0

    if True:
        with torch.no_grad():
            for batch_idx, (mfccs_padded, tokens_padded, input_lengths, target_lengths) in enumerate(dev_dataloader):
                mfccs_padded = mfccs_padded.to(device)
                input_lengths = input_lengths.to(device)
                tokens_padded_cpu = tokens_padded.cpu()

                # log_probs = model(mfccs_padded, input_lengths)
                log_probs = model(mfccs_padded)
                # print("Log probabilities shape:", log_probs.shape) # Print shape
                # print("Log probabilities example (first utterance, first timestep):", log_probs[0, 0, :]) # Print probs for first timestep of first utterance
                
                # predicted_sentences = decode_predictions(log_probs, input_lengths)
                # predicted_sentences = ctc_decoder(log_probs, input_lengths)
                predicted_sentences = ctc_decoder_multiple(log_probs, input_lengths)
                # quit()
                # predicted_sentences = decode_predictions(log_probs, input_lengths, index_to_char, beam_width=3) # Or beam_width=5, etc.

                reference_sentences = []
                for tokens in tokens_padded_cpu:
                    sentence = "".join([index_to_char[token.item()] for token in tokens if token.item() != 0])
                    reference_sentences.append(sentence)
                
                # print('Predicted:', predicted_sentences)
                # print('Reference:', reference_sentences)
                
                # Print them one by one
                for i in range(len(predicted_sentences)):
                    if i > len(reference_sentences) - 1:
                        break
                    # print(f">>> Predicted: {predicted_sentences[i]}")
                    print(f">>> Predicted:")
                    for _ in range(len(predicted_sentences[i])):
                        print(">", predicted_sentences[i][_])
                    print(f">>> Reference: {reference_sentences[i]}")
                    print("")
                    input("Press Enter to continue...")

                batch_wer = calculate_wer(reference_sentences, predicted_sentences)
                if batch_wer is not None:
                    total_wer += batch_wer * len(reference_sentences)
                    num_utterances += len(reference_sentences)

        avg_wer = total_wer / num_utterances
        print(f"Validation WER: {avg_wer:.4f}")
        
if __name__ == '__main__':
    main()
    evaluate()

