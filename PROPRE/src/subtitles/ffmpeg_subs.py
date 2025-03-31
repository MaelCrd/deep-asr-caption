import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ffmpeg.ffmpeg_utils import run_ffmpeg_with_progress


def add_subtitles_to_video(video_path: str, subtitle_path: str, output_path: str) -> None:
    """
    Add subtitles to a video using ffmpeg.
    """
    # command = [
    #     'ffmpeg',
    #     # '-hwaccel', 'cuda',
    #     '-y',  # Overwrite output files without asking
    #     '-i', video_path,
    #     '-vf', f'subtitles={subtitle_path}',
    #     '-c:v', 'libx264', '-preset', 'ultrafast',
    #     # '-threads', '8',
    #     # '-c:a', 'copy',
    #     output_path
    # ]
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,
        '-i', subtitle_path,
        # '-map', '0:v', '-map', '1:s',
        '-c', 'copy',
        '-c:s', 'mov_text',
        '-metadata:s:s:0', "title=English (AI generated)",
        '-metadata:s:s:0', "language=eng",
        # '-c:v', 'libx264', '-preset', 'ultrafast',
        output_path
    ]
    
    # Execute the command
    # subprocess.run(command, check=True)
    run_ffmpeg_with_progress(command)
    # print(f"Subtitles added to {video_path} and saved to {output_path}.")


if __name__ == "__main__":
    video_path = "data/input/" + "youtube_1B3B_LLM.mp4"  # Path to the input video file
    subtitle_path = "example.srt"  # Path to the subtitle file
    output_path = "data/output/" + "output_video.mp4"  # Path to save the output video with subtitles

    add_subtitles_to_video(video_path, subtitle_path, output_path)