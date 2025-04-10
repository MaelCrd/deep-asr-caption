import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ffmpeg.ffmpeg_utils import run_ffmpeg_with_progress


def add_subtitles_to_video(video_path: str, subtitle_path: str, output_path: str) -> None:
    """
    Add subtitles to a video using ffmpeg.
    """
    # Get all subtitles files starting with 'subtitle_path'
    subtitle_files = []
    for file in os.listdir(os.path.dirname(subtitle_path)):
        if file.startswith(os.path.basename(subtitle_path)):
            subtitle_files.append(os.path.abspath(os.path.join(os.path.dirname(subtitle_path), file)))
    print(subtitle_files)
    
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', f'{video_path}',
    ]#     inputs,
    #     '-map 0',
    #     maps,
    #     '-c', 'copy',
    #     '-c:s', 'mov_text',
    #     # '-metadata:s:s:0', "title=English (AI generated)",
    #     # '-metadata:s:s:0', "language=eng",
    #     metadata,
    #     output_path
    # ]
    
    langages_codes = {'en': 'eng', 'fr': 'fra', 'es': 'spa'}
    langages_titles = {'en': 'English', 'fr': 'French', 'es': 'Spanish'}
    
    metadata = []
    for i in range(len(subtitle_files)):
        lang = subtitle_files[i].split('_')[-1].split('.')[0]
        if lang in langages_codes:
            lang_code = langages_codes[lang]
            lang_title = langages_titles[lang]
            metadata.append((lang_code, lang_title))
    
    for file in subtitle_files:
        command.append("-i")
        command.append(f'{file}')
    
    command.append('-map')  # Map the first input (video)
    command.append('0')     # Video stream from the first input
    
    for i in range(1, len(subtitle_files) + 1):
        command.append('-map')
        command.append(f'{i}')
    
    command.append('-c')
    command.append('copy')  # Copy the video and audio streams without re-encoding
    command.append('-c:s')
    command.append('mov_text')  # Use mov_text codec for subtitles
    
    for i, (lc, lt) in enumerate(metadata):
        command.append(f'-metadata:s:s:{i}')
        command.append(f'language={lc}')
        command.append(f'-metadata:s:s:{i}')
        command.append(f'title={lt} (AI generated)')
    
    command.append(f'{output_path}')  # Output file
    
    # print(f"'{' '.join(command)}'")
    
    # Execute the command
    # subprocess.run(command, check=True)
    run_ffmpeg_with_progress(command)
    # print(f"Subtitles added to {video_path} and saved to {output_path}.")


if __name__ == "__main__":
    video_path = "data/input/" + "youtube_1B3B_LLM.mp4"  # Path to the input video file
    subtitle_path = "example.srt"  # Path to the subtitle file
    output_path = "data/output/" + "output_video.mp4"  # Path to save the output video with subtitles

    add_subtitles_to_video(video_path, subtitle_path, output_path)