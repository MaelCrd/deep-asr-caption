import re
from tqdm import tqdm
import subprocess


def run_ffmpeg_with_progress(command):
    """
    Run ffmpeg command and display progress in the console using tqdm.
    """
    process = subprocess.Popen(
        command,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding='utf-8'  # Explicitly set encoding to UTF-8
    )

    # Regular expressions to match the progress and duration information
    progress_regex = re.compile(r'time=(\d+:\d+:\d+.\d+)')
    duration_regex = re.compile(r'Duration: (\d+:\d+:\d+.\d+),')

    total_duration = None

    # Initialize tqdm progress bar
    progress_bar = None

    for line in process.stderr:
        if total_duration is None:
            duration_match = duration_regex.search(line)
            if duration_match:
                total_duration = _convert_to_seconds(duration_match.group(1))
                progress_bar = tqdm(total=total_duration, unit='s')

        if progress_bar:
            progress_match = progress_regex.search(line)
            if progress_match:
                current_time = _convert_to_seconds(progress_match.group(1))
                progress_bar.n = current_time
                progress_bar.refresh()

    process.wait()
    if progress_bar:
        progress_bar.close()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def _convert_to_seconds(time_str):
    """Convert a time string in the format HH:MM:SS.xxx to seconds."""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s
