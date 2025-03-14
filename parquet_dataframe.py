import soundfile as sf
import io

count = 0

# Function to convert audio data to ndarray
def audio_to_ndarray(row, df_len):
    global count
    if count % 200 == 0: print(count, "/", df_len, end="\r")
    count += 1
    audio_data = row['audio']['bytes']
    audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
    return audio_array

def dataframe_from_parquet(filename):
    import pandas as pd
    df = (
        pd.read_parquet(filename, columns=["id", "text", "audio", "words", "word_start", "word_end"])
        .set_index("id")
    )
    
    # Apply the function to each row in the dataframe and create a new column with the ndarray
    # doing this makes the audio data much bigger, so we will not do this for now
    # df['audio'] = df.apply(audio_to_ndarray, axis=1, df_len=len(df.index))
    
    # return the dataframe
    return df


import soundfile as sf
import numpy as np
import io

def audio_bytes_to_ndarray(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_io)
    return audio_data