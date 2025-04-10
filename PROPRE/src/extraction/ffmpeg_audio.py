import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ffmpeg.ffmpeg_utils import run_ffmpeg_with_progress

# >>>>>>> Mettre dans le script principal
# _DATA_PATH = "data/"
# INPUT_PATH = _DATA_PATH + "input/"
# OUTPUT_PATH = _DATA_PATH + "output/"


def extract_audio_from_video(video_path: str, audio_path: str, progress_callback=None) -> float:
    """
    Extract audio from a video file using ffmpeg.
    Returns the duration of the audio in seconds.
    """
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_path
    ]
    
    # Execute the command
    # subprocess.run(command, check=True)
    duration = run_ffmpeg_with_progress(command, progress_callback)
    # print(f"Audio extracted from {video_path} and saved to {audio_path}.")
    
    return duration  # Get the duration of the audio file
    

if __name__ == "__main__":
    video_path = "data/input/" + "youtube_1B3B_LLM.mp4"  # Path to the input video file
    audio_path = "data/output/" + "youtube_1B3B_LLM.mp3"  # Path to save the extracted audio

    extract_audio_from_video(video_path, audio_path)