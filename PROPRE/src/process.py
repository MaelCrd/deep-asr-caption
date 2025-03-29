# Preprocessing imports
import extraction.ffmpeg_audio as ffmpeg_audio

# Prediction imports
import prediction.model_predict as model_predict

# Postprocessing imports
import postprocessing.A_groupby as A_groupby
import postprocessing.B_ollama_correct as B_ollama_correct
import postprocessing.B_spellchecking as B_spellchecking
import postprocessing.C_correlation as C_correlation
import postprocessing.D_partitioning as D_partitioning

# Subtitle imports
import subtitles.ffmpeg_subs as ffmpeg_subs
import subtitles.subs_file as subs_file

# Paths
_DATA_PATH = "data/"
_INPUT_PATH = _DATA_PATH + "input/"
INPUT_VIDEO_PATH = _INPUT_PATH
INPUT_AUDIO_PATH = _INPUT_PATH + "audios/"
_OUTPUT_PATH = _DATA_PATH + "output/"
OUTPUT_SUBS_PATH = _OUTPUT_PATH + "subtitles/"
OUTPUT_VIDEOS_PATH = _OUTPUT_PATH + "videos/"


def process(video_filename_or_path):
    """
    Process the video file by extracting audio, predicting transcription, post-processing, and adding subtitles to the video.\n
    Args:\n
        - video_filename (str): The name of the video file to process. Can also be a path.\n
    Returns:\n
        output_video_path (str): Path to the output video file with subtitles.\n
        subtitles_path (str): Path to the generated subtitle file.\n
    """
    
    # Extract video name (without path (/or \) and extension)
    video_name = video_filename_or_path.split("/")[-1] if '/' in video_filename_or_path else video_filename_or_path.split("\\")[-1]  # Extract the video name from the path
    id = ''.join(video_name.split(".")[:-1]) # ID for the video, used for naming output files
    
    # Input video path
    if '/' in video_filename_or_path or '\\' in video_filename_or_path:
        video_path = video_filename_or_path
    else:
        video_path = INPUT_VIDEO_PATH + video_filename_or_path  # Path to the input video file

    # Output audio path
    audio_path = INPUT_AUDIO_PATH + id + ".mp3"  # Path to save the extracted audio

    # Extract audio from video
    print(f"Extracting audio from {video_path}...")
    real_duration = ffmpeg_audio.extract_audio_from_video(video_path, audio_path)

    # # Prediction (mocked for now)
    # predicted = "----------------------s-o----mee--  -a-------s--s--i--d----  -i----ss--- --o---f--tt-enn-- --a----d--d-e-d--  two-----------  -i-nn----hhi-----b-b-i--t-------  c--ll-i-mm-err--i-s---a----t---iio--n------  -i---n---  the-  two-----------be----------------------------------------------------.  -i-f-- yyou--  w-a-nnt  -a----  ssp-e-eed  u---p---- the- ss-e-t-t-ing  of- ss-i---p-er--l--uu-------------------, --o---nee  w-ay-----  -i-s--  t-o- hha----dd-  m-o--rre-  -n-eg-ggedt-of  -i---------o-----n-----ss-----, the---------  -inn---i-t----ii----a---t-orrss--------------- tha-t-------  ssttaa-rr--t--- the----- p----ll-e-mmerr--i--s---a--t---iio-n---------- rr-e----a----c--t--iionn-----------------------.  y-ou--- c-ann---------  b-y-----------  -ex---c-e---lllerrraa--t-orr--ss----  sp-e----s-i---f-i-cal-lly-- f-or- thiss  -b-ur---p-use- -of-f-  the  shell-------f-----------------------------."
    # total_time = len(predicted) # Example total time, no units
    # real_duration = 4.3  # Example duration in seconds
    
    # Predict the transcription
    print(f"Predicting transcription for audio file '{audio_path}'...")
    predicted = model_predict.predict(audio_path)  # Predict the transcription
    total_time = len(predicted)  # Total time, no units
    print(f"Predicted transcription: {predicted}")
    
    # # Postprocessing steps
    # min_chars = 20
    # max_chars = 60

    # A_groupby: Group by words
    sentence, start_indices = A_groupby.group_and_timestamps(predicted)
    print(f"Grouped sentence: {sentence}")
    
    # B_spellchecking: Correct subtitles using spellchecking
    print(f"Correcting sentence using spellchecking...")
    corrected_sentence = B_spellchecking.correct_sentence(sentence)
    print(f"Corrected sentence (spellchecking): {corrected_sentence}")

    # B_ollama_correct: Correct subtitles using Ollama
    # print(f"Correcting sentence...")
    # corrected_sentence = B_ollama_correct.correct_sentence(sentence)
    # corrected_sentence = sentence  # Mocked for now, replace with actual correction
    # print(f"Corrected sentence (ollama): {corrected_sentence}")

    # C_correlation: Correlate subtitles with timestamps
    ajusted_timestamps = C_correlation.correlate_sequences_with_timestamps(sentence, start_indices, corrected_sentence)

    # D_partitioning: Partition subtitles into final format
    subtitles, subs_timestamps, error = D_partitioning.partition_transcription(corrected_sentence, ajusted_timestamps)#, min_chars, max_chars)
    if error:
        print(f">>> Error: {error}")
    print(f"Subtitles: {subtitles}")

    # Create subtitle file
    subtitles_path = OUTPUT_SUBS_PATH + id + ".srt"  # Path to save the subtitle file
    subs_file.create_subtitle_file(subtitles, subs_timestamps, total_time, real_duration, subtitles_path)

    # Add subtitles to video
    output_video_path = OUTPUT_VIDEOS_PATH + id + "_subs.mp4"  # Path to save the output video with subtitles
    print(f"Adding subtitles to video '{video_path}'...")
    ffmpeg_subs.add_subtitles_to_video(video_path, subtitles_path, output_video_path)
    
    return output_video_path, subtitles_path, "Subtitles added successfully."


if __name__ == "__main__":
    # input("Enter to start the process...")
    video_filename = "youtube_1B3B_LLM-26s.mp4"  # Name of the input video file
    # video_filename = "youtube_1B3B_LLM.mp4"  # Name of the input video file
    # video_filename = "youtube_Bt-7YiNBvLE_1920x1080_h264.mp4"  # Name of the input video file
    output_video_path, subtitles_path, message = process(video_filename)
    
