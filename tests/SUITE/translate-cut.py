import re

def split_translated_transcript_proportionally(
    english_transcript: str,
    english_segments: list[str],
    translated_transcript: str
) -> list[str]:
    """
    Splits a translated transcript into segments that proportionally match
    the word count distribution of the original English segments.

    Args:
        english_transcript: The full original English transcript (string).
        english_segments: A list of strings, where each string is a pre-split
                          segment of the original English transcript.
        translated_transcript: The full translated transcript (string).

    Returns:
        A list of strings representing the translated transcript split into
        segments, attempting to match the proportional length of the
        English segments. Returns empty segments if inputs are empty or
        cannot be processed.
    """

    # --- Input Validation ---
    if not english_transcript or not translated_transcript or not english_segments:
        print("Warning: One or more inputs are empty.")
        # Return a list of empty strings matching the number of expected segments
        return [""] * len(english_segments)

    # --- Tokenization (Split into words) ---
    # Using a simple split() for broad compatibility.
    # For more accuracy (handling punctuation), consider regex or NLTK:
    # english_words_full = re.findall(r'\b\w+\b', english_transcript.lower())
    # translated_words = re.findall(r'\b\w+\b', translated_transcript.lower())
    english_words_full = english_transcript.split()
    translated_words = translated_transcript.split()

    total_english_words = len(english_words_full)
    total_translated_words = len(translated_words)

    if total_english_words == 0 or total_translated_words == 0:
        print("Warning: Cannot process empty transcript text.")
        return [""] * len(english_segments)

    # --- Calculate Cumulative Word Counts for English Segments ---
    # This helps determine the proportion of the text covered *up to the end*
    # of each segment.
    cumulative_english_word_counts = []
    current_word_count = 0
    word_counts_in_segments = [] # Store individual counts for proportion calculation base

    for segment in english_segments:
        segment_words = segment.split()
        segment_word_count = len(segment_words)
        word_counts_in_segments.append(segment_word_count)
        current_word_count += segment_word_count
        cumulative_english_word_counts.append(current_word_count)

    # Use the sum of words *actually found in segments* as the denominator for proportions.
    # This handles cases where the provided segments might not perfectly match the full transcript.
    total_english_words_in_segments = sum(word_counts_in_segments)

    if total_english_words_in_segments == 0:
         print("Warning: English segments contain no words.")
         return [""] * len(english_segments)


    # --- Calculate Target Split Points in Translated Text ---
    # Find the target word index in the translated text corresponding to the
    # end of each English segment, based on cumulative proportion.
    translated_split_indices = []
    for english_cumulative_count in cumulative_english_word_counts:
        proportion = english_cumulative_count / total_english_words_in_segments
        # Round to nearest whole word index
        target_index = round(proportion * total_translated_words)
        # Ensure index doesn't exceed the total number of translated words
        target_index = min(target_index, total_translated_words)
        translated_split_indices.append(target_index)

    # --- Create Translated Segments ---
    translated_segments = []
    start_index = 0
    for end_index in translated_split_indices:
        # Ensure start_index is not greater than end_index (can happen with empty segments/rounding)
        actual_end_index = max(start_index, end_index)
        # Slice the translated words list
        segment_words = translated_words[start_index:actual_end_index]
        # Join the words back into a string segment
        translated_segments.append(" ".join(segment_words))
        # Update the start index for the next segment
        start_index = actual_end_index

    # --- Final Checks ---
    # Ensure the number of output segments matches the number of input English segments.
    # This might be needed if the last English segment was empty or due to rounding issues.
    while len(translated_segments) < len(english_segments):
        translated_segments.append("") # Add empty segments if needed

    # If rounding resulted in *more* segments (unlikely with cumulative approach, but safety check)
    if len(translated_segments) > len(english_segments):
        # Combine the extra segments into the last valid one.
        last_valid_segment_index = len(english_segments) - 1
        # Get the words from the start of the last valid segment onwards
        start_index_for_last = translated_split_indices[last_valid_segment_index - 1] if last_valid_segment_index > 0 else 0
        # All remaining words go into the true last segment
        last_segment_words = translated_words[start_index_for_last:]
        translated_segments = translated_segments[:last_valid_segment_index] # Trim excess
        translated_segments.append(" ".join(last_segment_words)) # Add corrected last segment

    return translated_segments

# --- Example Usage ---
eng_transcript = "The magini happen across a short mulyscript that describes a seene between a person, and their a i assistant. Thesgriptd has what the person asks the a eye, but the aeyes respons has been tourn off. Suppose you also have this powerful magical machine the cantake any text and provide a sensible prediction of what word comes next. They could then a finished thescript by feeding ind what you have to the machine, seeing what it would predict to start"
eng_segments = [
    'The magini happen across a short mulyscript that describes a seene between a person,',
    'and their a i assistant.',
    'Thesgriptd has what the person asks the a eye,',
    'but the aeyes respons has been tourn off.',
    'Suppose you also have this powerful magical machine the cantake any text and provide a',
    'sensible prediction of what word comes next.',
    'They could then a finished thescript by feeding ind what you have to the machine,',
    'seeing what it would predict to start'
]
fr_transcript = "Les magini se produisent à travers un court mulyscrit qui décrit une vue entre une personne, et leur assistante. Thesgriptd a ce que la personne demande à l'œil, mais la responsabilité des yeux a été tournée. Supposons que vous avez aussi cette machine magique puissante le cantake n'importe quel texte et fournir une prédiction raisonnable de ce mot vient ensuite. Ils pourraient alors un script fini en nourrissant ce que vous avez à la machine, voir ce qu'il pourrait prédire pour commencer"

translated_splits = split_translated_transcript_proportionally(
    eng_transcript, eng_segments, fr_transcript
)

print(f"Original English Segments ({len(eng_segments)}):")
for i, seg in enumerate(eng_segments):
    print(f"{i}: {seg}")

print("\n---")

print(f"Proportionally Split Translated Segments ({len(translated_splits)}):")
for i, seg in enumerate(translated_splits):
    print(f"{i}: {seg}")