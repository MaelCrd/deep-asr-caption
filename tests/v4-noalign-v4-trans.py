import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MFCC
import jiwer  # For WER calculation
from torchaudio.transforms import Resample  # For audio resampling
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Import functional
from datetime import datetime
import soundfile as sf
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import parquet_dataframe

if __name__ == '__main__':
    # 3. Data Preprocessing
    # 3.1 Audio Feature Extraction (MFCCs)
    n_mfcc = 20
    sample_rate = 16000
    mfcc_transform = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 22}
    )

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
        def __init__(self, dataset, mfcc_transform, tokenizer):
            self.dataset = dataset
            self.mfcc_transform = mfcc_transform
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # audio, sample_rate, transcript, _, _, _ = self.dataset[idx]
            # mfccs_untransposed = self.mfcc_transform(resample(audio, sample_rate, 16000)) # Keep untransposed for length
            # mfccs = mfccs_untransposed.transpose(0, 1) # THEN transpose
            # tokens = self.tokenizer(transcript)
            # return mfccs, torch.tensor(tokens), mfccs_untransposed # Return untransposed MFCC for length
            
            audio_mfccs_untransposed, mfccs, transcript = self.dataset[idx]
            tokens = self.tokenizer(transcript)
            return mfccs, torch.tensor(tokens), audio_mfccs_untransposed # Return untransposed MFCC for length

    def collate_fn_asr(batch):
        mfccs_batch = [item[0] for item in batch] # These are already transposed
        tokens_batch = [item[1] for item in batch]
        mfccs_untransposed_batch = [item[2] for item in batch] # Get untransposed MFCCs

        mfccs_padded = pad_sequence(mfccs_batch, batch_first=True)
        tokens_padded = pad_sequence(tokens_batch, batch_first=True, padding_value=0)
        input_lengths = torch.tensor([mfcc.shape[1] for mfcc in mfccs_untransposed_batch]) # Use untransposed for length (shape[1] is time_frames)
        target_lengths = torch.tensor([len(tokens) for tokens in tokens_batch])

        # print("--- Batch Data Inspection ---") # Keep print statements for now to verify
        # print("MFCCs padded shape:", mfccs_padded.shape)
        # print("Tokens padded shape:", tokens_padded.shape)
        # print("Input lengths:", input_lengths)
        # print("Target lengths:", target_lengths)
        # print("MFCCs batch min/max:", mfccs_padded.min(), mfccs_padded.max())
        # if torch.isnan(mfccs_padded).any() or torch.isinf(mfccs_padded).any():
        #     print("WARNING: NaN or Inf values in MFCCs_padded!")
        # print("Tokens batch min/max:", tokens_padded.min(), tokens_padded.max())
        # if torch.isnan(tokens_padded).any() or torch.isinf(tokens_padded).any():
        #     print("WARNING: NaN or Inf values in tokens_padded!")

        return mfccs_padded, tokens_padded, input_lengths, target_lengths


    # # parquet_filename = "valid-00000-of-00001.parquet" #
    # parquet_filename = "train-00000-of-00064.parquet" #
    # parquet_filename = "train-00001-of-00064.parquet" #
    # parquet_filename = "train-00063-of-00064.parquet" #

    files_source = []
    files_source.append("train-00000-of-00064.parquet")
    files_source.append("train-00001-of-00064.parquet")
    files_source.append("train-00002-of-00064.parquet")
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
    
    # files_source.append("train-00000-of-00044.parquet") #
    # files_source.append("train-00001-of-00044.parquet")
    # files_source.append("train-00002-of-00044.parquet")
    # files_source.append("train-00003-of-00044.parquet")
    # files_source.append("train-00004-of-00044.parquet")
    # files_source.append("train-00005-of-00044.parquet")

    # for i in range(0, 0+1):
    #     for end in ['44', '56', '64', '1033']:
    #         files_source.append(f"train-{str(i).zfill(5)}-of-{str(end).zfill(5)}.parquet")

    # files_source.append("train-00011-of-00064.parquet")
    # files_source.append("train-00012-of-00064.parquet")

    # files_source.append("valid-00000-of-00001.parquet")
    
    # files_source.append("train-00000-of-00056.parquet")

    print("Files source:", files_source)

    # import multiprocessing as mp
    # from functools import partial
    

    dataset = []
    for filename in files_source:
        filename = "dataset/" + filename
        print("Processing file:", filename)
        dataframe = parquet_dataframe.dataframe_from_parquet(filename) # Load a subset for testing

        print("Dataframe shape:", dataframe.shape)

        for i in range(0, len(dataframe)):
            mfccs_untransposed = mfcc_transform(torch.tensor(np.array(parquet_dataframe.audio_bytes_to_ndarray(dataframe.iloc[i]['audio']['bytes']))).float())
            dataset.append([
                mfccs_untransposed, # mfccs untransposed
                mfccs_untransposed.transpose(0, 1), # mfccs transposed
                dataframe.iloc[i]['text'], # transcript
            ])
            
            # # Save to file the audio
            # audio_path = f"audio_{i}.wav"
            # sf.write(audio_path, np.array(parquet_dataframe.audio_bytes_to_ndarray(dataframe.iloc[i]['audio']['bytes'])), 16000)
            # with open(f"audio_{i}_transcript.txt", "w") as f:
            #     f.write(dataframe.iloc[i]['text'])
            
            # if i > 400:
            #     break
        
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
        # print(log_probs.shape)
        predicted_tokens = torch.argmax(log_probs, dim=2).cpu().numpy()
        
        decoded_sentences = []
        for i in range(predicted_tokens.shape[0]):
            tokens = predicted_tokens[i, :input_lengths[i]]
            # use groupby to find continuous same indexes
            # tokens = [k for k, g in groupby(tokens)]
            sentence = ""
            for t in tokens:
                char = index_to_char[t.item()]
                # if char != '<blank>': # Assuming no blank token in this simple example, adjust if needed
                #     sentence += char
                sentence += char if char != '<blank>' else '-'
            decoded_sentences.append(sentence)
        return decoded_sentences
    
    def ctc_decoder_predict(log_probs):
        # print(log_probs.shape)
        predicted_tokens = torch.argmax(log_probs, dim=2).cpu().numpy()
        # print(predicted_tokens.shape)
        # print(predicted_tokens)
        
        decoded_sentences = []
        for i in range(1):
            tokens = predicted_tokens[i, :]
            # use groupby to find continuous same indexes
            # tokens = [k for k, g in groupby(tokens)]
            sentence = ""
            for t in tokens:
                char = index_to_char[t.item()]
                # if char != '<blank>': # Assuming no blank token in this simple example, adjust if needed
                #     sentence += char
                sentence += char if char != '<blank>' else '-'
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
    class TransformerASR(nn.Module):
        def __init__(self, input_dim, output_dim, d_model=64, nhead=8, num_encoder_layers=3, dropout=0.2):
            super(TransformerASR, self).__init__()
            self.input_dim = input_dim # MFCC features
            self.output_dim = output_dim # Character or subword vocab size

            self.embedding = nn.Linear(input_dim, d_model) # Project MFCCs to transformer's d_model

            # Transformer Encoder Layers
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout, batch_first=True)
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

            self.output_layer = nn.Linear(d_model, output_dim + 1) # +1 for blank token
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, input_lengths): # added input_lengths
            # x shape: (batch_size, time_steps, input_dim) - > (32, 842, 20)
            x = self.embedding(x) # (batch_size, time_steps, d_model) -> (32, 842, 256)

            # Create a mask for padding (crucial for transformers)
            mask = torch.arange(x.size(1)).unsqueeze(0).to(device) >= input_lengths.unsqueeze(1).to(device) # (batch_size, time_steps)
            mask = mask.to(x.device) # Move mask to the same device as x

            # Transformer encoder expects source mask
            x = self.transformer_encoder(x, src_key_padding_mask=mask) # (batch_size, time_steps, d_model) -> (32, 842, 256)
            x = self.dropout(x)
            x = self.output_layer(x) # (batch_size, time_steps, output_dim + 1) -> (32, 842, 256)
            x = F.log_softmax(x, dim=-1)
            return x

    input_dim = n_mfcc
    # hidden_dim = 512 # big : 810
    output_dim = len(characters)
    model = TransformerASR(input_dim, output_dim)
    print(model)

    # 5. Loss Function and Optimizer (CTC Loss and Adam)
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # 6. Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion.to(device)

    TRAIN = True # Set to True to train the model

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
                    # log_probs = model(mfccs_padded)
                    log_probs = model(mfccs_padded, input_lengths)  # Pass input_lengths for masking
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

                    batch_interval = 5
                    if batch_idx % batch_interval == 0 and batch_idx > 0:
                        # avg_loss = train_loss / (batch_idx + 1)
                        avg_loss = batch_group_loss / batch_interval # Print average loss over last few batches
                        batch_group_loss = 0.0
                        print(f"[{datetime.now().strftime("%d-%m %H:%M")}] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Train Loss (last batch group): {avg_loss:.4f}")
                        # torch.save(model.state_dict(), "asr_v4_trans.pth")
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
                        # log_probs = model(mfccs_padded)
                        log_probs = model(mfccs_padded, input_lengths)  # Pass input_lengths for masking
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
                torch.save(model.state_dict(), "asr_v4_trans.pth")

            print("Training finished!")

            # Save model checkpoint
            torch.save(model.state_dict(), "asr_v4_trans.pth")
        else :
            model.load_state_dict(torch.load("asr_v4_trans.pth"))
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
                    predicted_sentences = ctc_decoder(log_probs, input_lengths)
                    # predicted_sentences = ctc_decoder_multiple(log_probs, input_lengths)
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
                        print(f">>> Predicted: {predicted_sentences[i]}")
                        # print(f">>> Predicted:")
                        # for _ in range(len(predicted_sentences[i])):
                        #     print(">", predicted_sentences[i][_])
                        # print(">", predicted_sentences[i])
                        print(f">>> Reference: {reference_sentences[i]}")
                        print("")
                        input("Press Enter to continue...")

                    batch_wer = calculate_wer(reference_sentences, predicted_sentences)
                    if batch_wer is not None:
                        total_wer += batch_wer * len(reference_sentences)
                        num_utterances += len(reference_sentences)

            avg_wer = total_wer / num_utterances
            print(f"Validation WER: {avg_wer:.4f}")
    
    def predict(audio_path):
        model.eval()
        
        print(f"Predicting for audio file '{audio_path}'...")

        # Load an example audio file
        audio, sample_rate = sf.read(audio_path)
        # print("Audio shape:", audio.shape)
        # print("Sample rate:", sample_rate)
        
        # Make sure audio is mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(torch.tensor(audio).float()).numpy()
            sample_rate = 16000
        
        # Cut audio to X seconds
        audio = audio[:1300*sample_rate]
        
        audio_transcript_path = audio_path.replace(".wav", "_transcript.txt")
        try:
            with open(audio_transcript_path, "r") as f:
                transcript = f.read()
        except Exception:
            transcript = "?"
        
        # Extract MFCCs
        audio_tensor = torch.tensor(audio).float()
        mfccs_untransposed = mfcc_transform(audio_tensor)
        mfccs = mfccs_untransposed.transpose(0, 1)
        # print("MFCCs shape:", mfccs.shape)
        
        # Add batch dimension
        mfccs = mfccs.unsqueeze(0)
        # print("MFCCs shape with batch dimension:", mfccs.shape)
        
        # Predict
        with torch.no_grad():
            mfccs = mfccs.to(device).contiguous()
            # log_probs = model(mfccs, torch.tensor([mfccs.shape[1]]).to(device))
            log_probs = model(mfccs)
            # log_probs = log_probs.transpose(0, 1)
            # print("Log probabilities shape:", log_probs.shape) # Print shape
            # Decode
            predicted_sentences = ctc_decoder_predict(log_probs) #,torch.tensor([mfccs.shape[1]])
            print("Predicted:", predicted_sentences[0])
            print("Reference:", transcript)
        
    main()
    # evaluate()
    
    audio_files = [
        # "audio_0.wav",
        # "audio_1.wav",
        # "audio_2.wav",
        # "audio_3.wav",
        # "audio_4.wav",
        # "audio_5.wav",
        "audio_6.wav",
        "there-is-this-famous-google-interview-question-that-everyone-gets-wrong.wav",
        # "peel-off.mp3",
        # "peel-off-edit.wav",
        # "off-the-shelf.mp3",
        "youtube_Ni82f1-cAXg_audio_glue.mp3",
    ]
    for af in audio_files:
        predict(af)
    
    # from pyinstrument import Profiler
    # with Profiler(interval=0.001) as profiler:
    #     main()
    # profiler.print()
    # profiler.output_html()

