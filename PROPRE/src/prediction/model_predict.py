import torch
import torch.nn as nn
from torchaudio.transforms import MFCC
from torchaudio.transforms import Resample  # For audio resampling
import torch.nn.functional as F  # Import functional
import soundfile as sf


# Audio feature extraction
N_MFCC = 20  # Number of MFCC features

# Text character set
characters = ["<blank>", " ", "'", ".", ",", "!", "?", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}


def _ctc_decoder_predict(log_probs):
    predicted_tokens = torch.argmax(log_probs, dim=2).cpu().numpy()
    
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


# Model Definition
class MediumASR(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(MediumASR, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Reshape for RNN
        x = x.permute(0, 2, 1, 3)  # Swap feature and channel dimensions
        # print("x.shape after permute:", x.shape)  # torch.Size([32, 842, 32, 5])
        batch_size, time_steps, channels, features = x.size()
        lstm_input_size = channels * features  # 32 * 5 = 160
        x = x.reshape(batch_size, time_steps, lstm_input_size)

        # RNN Layers
        x = x.contiguous()  # Make the tensor contiguous before the first LSTM
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

    def predict(self, audio_path):
        # Audio feature extraction
        SAMPLE_RATE = 16000
        mfcc_transform = MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 22}
        )

        # Load an example audio file
        audio, sample_rate = sf.read(audio_path)
        
        # Make sure audio is mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Resample to 16 kHz if needed
        if sample_rate != SAMPLE_RATE:
            resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            audio = resampler(torch.tensor(audio).float()).numpy()
            sample_rate = SAMPLE_RATE
        
        # Cut audio to X seconds (temporary fix for long audio files)
        audio = audio[:1300*sample_rate]
        
        # Transcript is not available in this example
        transcript = "?"
        
        # Extract MFCCs
        audio_tensor = torch.tensor(audio).float()
        mfccs_untransposed = mfcc_transform(audio_tensor)
        mfccs = mfccs_untransposed.transpose(0, 1)
        
        # Add batch dimension
        mfccs = mfccs.unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            mfccs = mfccs.to(self.device).contiguous()
            log_probs = self(mfccs)
            # Decode
            predicted_sentences = _ctc_decoder_predict(log_probs)
            # print("Predicted:", predicted_sentences[0])
            # print("Reference:", transcript)
        
        return predicted_sentences[0]


def _load_model(model_path):
    # Load the model
    input_dim = N_MFCC
    output_dim = len(characters)
    model = MediumASR(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Set device
    model.to(model.device)
    print(f"Using device: {model.device}")
    return model


# Load the model
MODEL = _load_model("data/models/7_asr_v3.pth")


def predict(audio_path):
    """
    Predict the transcription of an audio file.
    """
    return MODEL.predict(audio_path)


if __name__ == '__main__':
    print(MODEL)
    # input("Press Enter to continue...")
    
    audio_files = [
        # "audio_0.wav",
        # "audio_1.wav",
        # "audio_2.wav",
        # "audio_3.wav",
        # "audio_4.wav",
        # "audio_5.wav",
        # "audio_6.wav",
        # "there-is-this-famous-google-interview-question-that-everyone-gets-wrong.wav",
        "peel-off.mp3",
        # "peel-off-edit.wav",
        "off-the-shelf.mp3",
        # "youtube_Ni82f1-cAXg_audio_glue.mp3",
        # "god-call.mp3",
        # "youtube_AaCnBOqyvIM_audio_fireship_exploit.mp3",
    ]
    
    for af in audio_files:
        audio_path = "data/output/audios/" + af
        print(f"Predicting for audio file '{audio_path}'...")
        
        # Predict the transcription
        predicted = predict(audio_path)
        
        # Print the predicted transcription
        print(f"Predicted transcription: {predicted}")
    

