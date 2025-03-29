import pysubs2


def timestamps_to_milliseconds(timestamps: list[tuple[int,int]], total_time: int, real_duration: float) -> list[tuple[float, float]]:
    """
    Convert timestamps from integers to milliseconds based on the total time and real duration.\n
    Ex:\n
    Input :\n
        timestamps = [(22, 219), (294, 369), (410, 483), (514, 661), (711, 858)]
        total_time = 894
        real_duration = 10.0 (seconds)\n
    Output :\n
        timestamps = [(0.2460, 2.4496), (3.2885, 4.1275), (4.5861, 5.4026), (5.7494, 7.3937), (7.9530, 9.5973)]\n
    """
    new_timestamps = []
    for start, end in timestamps:
        new_start = start * real_duration * 1000 / total_time
        new_end = end * real_duration * 1000 / total_time
        new_timestamps.append((new_start, new_end))
    return new_timestamps


def create_subtitle_file(subtitles: list[str], timestamps: list[tuple[int,int]], total_time: int, real_duration: float, output_path: str) -> None:
    """
    Create a subtitle file from the subtitles and timestamps.
    """
    subs = pysubs2.SSAFile()
    subs.info['Title'] = 'Generated Subtitles'
    
    timestamps_sec = timestamps_to_milliseconds(timestamps, total_time, real_duration)
    
    # Add a small delay to the last word to ensure subtitles are readable
    # This can be cut down if the last word is not the last subtitle
    last_word_additional_time = 0.5  # seconds
    
    for i, sub in enumerate(subtitles):
        start_time, end_time = timestamps_sec[i]
        next_start_time = 9e99 # Default to a very large number
        if i + 1 < len(timestamps_sec):
            next_start_time = timestamps_sec[i + 1][0] - 1 # 1 millisecond before the next start time
        # Take the minimum of the next start time and the end time + last word additional time
        # This ensures that the subtitle does not overlap with the next one
        next_start_time = min(end_time + last_word_additional_time * 1000, next_start_time)
        
        # print(f"Start: {start_time:.3f}, End: {end_time:.3f}, Subtitle: {sub}")
        # Create a new subtitle event
        event = pysubs2.SSAEvent(
            start=start_time,
            end=next_start_time,
            text=sub
        )
        # Append the event to the subtitle file
        subs.append(event)
    
    subs.save(output_path)


if __name__ == "__main__":
    # Example usage
    subtitles = ['Some acid is often added to inhibit polymerization in the tube.', 'If you want to speed up the setting of superglue,', 'one way is to add more negative ions,', 'the initiators that start the polymerization reaction.', 'You can buy accelerators specifically for this purpose off the shelf.']
    timestamps = [(22, 219), (294, 369), (410, 483), (514, 661), (711, 858)]
    total_time = 894
    real_duration = 10.0  # Example duration in seconds
    
    # # Convert timestamps to seconds
    # timestamps = timestamps_to_seconds(timestamps, total_time, real_duration)
    # print("Converted timestamps:")
    # print(timestamps)
    # for start, end in timestamps:
    #     print(f"Start: {start:.3f}, End: {end:.3f}")
        
    # Create subtitle file
    output_path = "example.srt"
    create_subtitle_file(subtitles, timestamps, total_time, real_duration, output_path)
    