# Description: Partition the transcription into subtitles of a certain length

def _capitalize_transcription(transcription):
    """
    Capitalize the first letter of each sentence in the transcription.
    """
    sentences = transcription.split('. ')
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    return '. '.join(capitalized_sentences)


def _transcription_to_subtitles(transcription, timestamps, min_chars, max_chars):
    """
    Split the transcription into subtitles of a certain length
    """
    subtitles = []
    sub_timestamps = []
    current_sub = ''
    transcription_words = transcription.split()
    for word in transcription_words:
        if len(current_sub) > min_chars and current_sub.strip()[-1] in ',.!?':
            subtitles.append(current_sub)
            current_sub = ''
        if len(current_sub) + len(word) < max_chars:
            current_sub += word + ' '
        else:
            subtitles.append(current_sub)
            current_sub = word + ' '
    if current_sub:
        subtitles.append(current_sub)
    # Remove empty subtitles and strip whitespace
    subtitles = [sub.strip() for sub in subtitles]
    
    # Apply timestamps
    total_words = 0
    for i in range(len(subtitles)):
        ts = [timestamps[total_words], ]
        total_words += len(subtitles[i].split())
        ts.append(timestamps[total_words - 1])
        sub_timestamps.append(tuple(ts))
    
    # # Debug timestamps
    # for ts_start, ts_end in sub_timestamps:
    #     print(ts_start)
    #     print(subtitles[sub_timestamps.index((ts_start, ts_end))])
    #     print(transcription_words[timestamps.index(ts_start):timestamps.index(ts_start) + 3])
    #     print('')
    #     print(ts_end)
    #     print(subtitles[sub_timestamps.index((ts_start, ts_end))])
    #     print(transcription_words[timestamps.index(ts_end)])
    #     print('-----')
    
    return subtitles, sub_timestamps


def _check_lengths(subtitles, min_chars, max_chars):
    """
    Check that the subtitles are within the length limits
    """
    for i, sub in enumerate(subtitles):
        if len(sub) > max_chars:
            raise Exception(f'Subtitle {i} is too long: {len(sub)} characters')
        elif len(sub) < min_chars:
            raise Exception(f'Subtitle {i} is too short: {len(sub)} characters')

def _check_no_lost(subtitles: list[str], transcription: str):
    """
    Check that no text is lost in the partitioning
    """
    sub_text = ' '.join(subtitles)
    if sub_text.lower() != transcription.lower():
        print(sub_text)
        raise Exception('Lost text in subtitles')


def partition_transcription(transcription: str, timestamps: list[int], min_chars: int = 20, max_chars: int = 90) -> tuple[list[str], list[tuple[int,int]], Exception]:
    """
    Partition the transcription into subtitles of a certain length and calculate the timestamps.\n
    Ex:\n
    Input:\n
        Transcription : "Some acid is often added to inhibit polymerization in the tube. If you want to speed up the setting of superglue, one way is to add more negative ions, the initiators that start the polymerization reaction. You can buy accelerators specifically for this purpose off the shelf."\n
        Timestamps : [22, 37, 62, 75, 93, 110, 127, 160, 206, 216, 219, 294, 300, 308, 318, 325, 336, 346, 351, 365, 369, 410, 419, 431, 438, 443, 455, 467, 483, 514, 529, 575, 589, 605, 614, 661, 711, 719, 735, 752, 789, 817, 823, 831, 846, 853, 858]\n
        (Optional) Min chars : (default: 20)\n
        (Optional) Max chars : (default: 90)\n
    Output:\n
        Subtitles : ['Some acid is often added to inhibit polymerization in the tube.', 'If you want to speed up the setting of superglue,', 'one way is to add more negative ions,', 'the initiators that start the polymerization reaction.', 'You can buy accelerators specifically for this purpose off the shelf.']\n
        Sub timestamps : [(22, 219), (294, 369), (410, 483), (514, 661), (711, 858)]\n
        Error : _None_ if no text is lost and lengths are correct else _the error_
    """
    transcription_capitalized = _capitalize_transcription(transcription)
    subtitles, sub_timestamps = _transcription_to_subtitles(transcription_capitalized, timestamps, min_chars, max_chars)

    error = None
    try:
        _check_lengths(subtitles, min_chars, max_chars)
        _check_no_lost(subtitles, transcription)
    except Exception as e:
        print(e)
        error = e
    
    return subtitles, sub_timestamps, error


if __name__ == "__main__":
    # Example usage with your provided transcription:
    transcription = "Some acid is often added to inhibit polymerization in the tube. If you want to speed up the setting of superglue, one way is to add more negative ions, the initiators that start the polymerization reaction. You can buy accelerators specifically for this purpose off the shelf."
    timestamps = [22, 37, 62, 75, 93, 110, 127, 160, 206, 216, 219, 294, 300, 308, 318, 325, 336, 346, 351, 365, 369, 410, 419, 431, 438, 443, 455, 467, 483, 514, 529, 575, 589, 605, 614, 661, 711, 719, 735, 752, 789, 817, 823, 831, 846, 853, 858]
    
    print("Transcription:")
    print(transcription)
    print("Timestamps:")
    print(timestamps)
    
    subtitles, sub_timestamps, error = partition_transcription(transcription, timestamps)
    if error:
        print("Error:", error)
        
    print()
    print(subtitles)
    print(sub_timestamps)
    
    # # Print the subtitles with their lengths
    # for i, sub in enumerate(subtitles):
    #     print(f'[{str(i+1).zfill(2)}] ({len(sub)}) {sub}')
