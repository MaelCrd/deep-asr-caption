# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio

class AlignedAudioDataset(Dataset):
    def __init__(self, texts, audios, words_lists, starts, ends, processor, tokenizer):
        self.texts = texts
        self.audios = audios
        self.words_lists = words_lists
        self.word_starts = starts
        self.word_ends = ends
        self.processor = processor
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        # Obtenir l'audio
        audio = self.audios[idx]
        # Conversion au format torch si nécessaire
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Traiter l'audio pour extraire les features
        features = self.processor.process(audio)
        
        # Tokeniser le texte
        tokens = self.tokenizer.encode(self.texts[idx])
        
        # Récupérer les alignements
        words = self.words_lists[idx]
        starts = self.word_starts[idx]
        ends = self.word_ends[idx]
        
        # Construire un tenseur d'alignement (pour l'entraînement supervisé)
        # Ceci est simplifié - dans la pratique, il faudrait aligner les tokens avec les mots
        alignments = []
        for i, token in enumerate(tokens):
            # Associer chaque token à un timestamp approximatif
            # Cette partie nécessite une logique d'alignement token-mot
            # car vos alignements sont au niveau du mot mais les tokens peuvent être différents
            alignments.append((0.0, 0.0))  # Placeholder
        
        return {
            "features": features,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "feature_length": features.shape[1],
            "token_length": len(tokens),
            "words": words,
            "word_starts": starts,
            "word_ends": ends
        }

# %%
def train_with_alignment(model, batch, ctc_criterion, alignment_criterion, optimizer, device):
    model.train()
    
    # Préparation des données
    features = batch["features"].to(device)
    tokens = batch["tokens"].to(device)
    feature_lengths = batch["feature_length"]
    token_lengths = batch["token_length"]
    
    # Forward pass
    log_probs = model(features, feature_lengths)
    
    # Calcul de la perte CTC standard
    ctc_loss = ctc_criterion(
        log_probs.transpose(0, 1),
        tokens,
        feature_lengths,
        token_lengths
    )
    
    # Ici, vous pourriez ajouter une perte auxiliaire basée sur les alignements
    # Cette partie est complexe et dépend de la façon dont vous représentez les alignements
    # dans votre modèle
    alignment_loss = torch.tensor(0.0).to(device)  # Placeholder
    
    # Perte totale
    loss = ctc_loss + alignment_loss
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    return loss.item(), ctc_loss.item(), alignment_loss.item()

# %%
class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Transform to extract mel spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Normalization
        self.normalizer = torchaudio.transforms.AmplitudeToDB()
        
    def process(self, audio_waveform):
        # Convert to mono if necessary
        if audio_waveform.shape[0] > 1:
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
            
        # Extract mel features
        mel_spec = self.mel_spectrogram(audio_waveform)
        mel_spec = self.normalizer(mel_spec)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec

# %%
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {" ": 0}  # Blank token for CTC
        self.idx = 1
        
    def encode(self, text):
        result = []
        for char in text:
            if char not in self.vocab:
                self.vocab[char] = self.idx
                self.idx += 1
            result.append(self.vocab[char])
        return result
        
    def decode(self, tokens):
        # Inverse mapping
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([inv_vocab.get(t, "") for t in tokens])

# %%
# Exemple de chargement
texts = []
audios = []
words_lists = []
word_starts = []
word_ends = []

import parquet_dataframe

dataframe = parquet_dataframe.dataframe_from_parquet('valid-00000-of-00001.parquet')

# Supposons que vos données sont dans un DataFrame pandas
for index, row in dataframe.iterrows():
    texts.append(row['text'])
    audios.append(row['audio'])
    words_lists.append(row['words'])
    word_starts.append(row['word_start'])
    word_ends.append(row['word_end'])
    
# Using torchaudio
audio_processor = AudioProcessor(sample_rate=16000)

# # Or using a pre-trained processor
# from transformers import Wav2Vec2Processor
# audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Simple character tokenizer
tokenizer = SimpleTokenizer()

# # Or using a pre-trained subword tokenizer
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Création du dataset
dataset = AlignedAudioDataset(
    texts, audios, words_lists, word_starts, word_ends, 
    processor=audio_processor, tokenizer=tokenizer
)

# %%
# Initialisation des composants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Définir l'AudioProcessor
audio_processor = AudioProcessor(sample_rate=16000, n_mels=80, n_fft=400, hop_length=160)

# Définir le tokenizer (caractère par caractère pour commencer)
tokenizer = SimpleTokenizer()

# %%



