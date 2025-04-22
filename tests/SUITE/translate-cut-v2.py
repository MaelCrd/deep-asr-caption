import re

def split_translated_transcript_by_punctuation(
    english_segments: list[str],
    translated_transcript: str,
    punctuation_marks: str = ".,!?;:" # Punctuation to look for
) -> list[str] | None:
    """
    Splits a translated transcript based on matching punctuation found at the
    end of the corresponding English segments.

    Args:
        english_segments: A list of strings, where each string is a pre-split
                          segment of the original English transcript.
        translated_transcript: The full translated transcript (string).
        punctuation_marks: A string containing punctuation characters to consider
                           as segment terminators.

    Returns:
        A list of strings representing the translated transcript split based
        on matching punctuation.
        Returns None if the number of segments generated doesn't match, or if
        a required punctuation mark couldn't be found, indicating the
        assumption likely failed.
        Logs warnings if issues are encountered.
    """
    if not translated_transcript or not english_segments:
        print("Warning: Translated transcript or English segments list is empty.")
        return [""] * len(english_segments) # Return empty structure

    translated_segments = []
    search_start_index = 0
    last_split_point = 0
    num_english_segments = len(english_segments)

    # Iterate through segments that *should* define a split point (all except the last)
    for i in range(num_english_segments - 1):
        eng_segment = english_segments[i].strip() # Remove leading/trailing whitespace

        if not eng_segment:
            # If the English segment is empty, assume the corresponding
            # translated segment is also empty. Doesn't advance search index.
            print(f"Warning: English segment {i} is empty. Adding empty translated segment.")
            translated_segments.append("")
            # We don't advance last_split_point or search_start_index here
            # as no text was consumed in the translation for this empty segment.
            continue

        # Find the last character that is considered punctuation
        last_char = eng_segment[-1] if eng_segment else None

        if last_char in punctuation_marks:
            # Try to find this punctuation in the translated text *after* the last split
            # We add 1 to find indexes *after* the character itself if needed.
            # We search for the specific character `last_char`.
            try:
                # Find the first occurrence *at or after* the search start index
                found_index = translated_transcript.index(last_char, search_start_index)

                # The split point is *after* the found punctuation
                split_at = found_index + 1

                # Extract the segment
                segment_text = translated_transcript[last_split_point:split_at].strip()
                translated_segments.append(segment_text)

                # Update pointers for the next search
                last_split_point = split_at
                search_start_index = split_at # Start next search after this punctuation

            except ValueError:
                # .index() raises ValueError if not found
                print(f"Error: Punctuation '{last_char}' from end of English segment {i} "
                      f"was NOT found in the translated transcript after index {search_start_index}.")
                print("Cannot reliably split using punctuation matching. Aborting.")
                # Append the rest of the text as the last segment and return None to signal failure
                remaining_text = translated_transcript[last_split_point:].strip()
                if remaining_text:
                     translated_segments.append(remaining_text)
                # Pad with empty strings if needed, although count will be wrong
                while len(translated_segments) < num_english_segments:
                    translated_segments.append("")
                return None # Signal failure

        else:
            # The English segment did not end with recognized punctuation
            print(f"Warning: English segment {i} ('...{eng_segment[-10:]}') "
                  f"does not end with recognized punctuation ({punctuation_marks}).")
            print("Cannot use punctuation matching for this split. Aborting.")
            # Append the rest of the text as the last segment and return None to signal failure
            remaining_text = translated_transcript[last_split_point:].strip()
            if remaining_text:
                 translated_segments.append(remaining_text)
            # Pad with empty strings if needed
            while len(translated_segments) < num_english_segments:
                translated_segments.append("")
            return None # Signal failure

    # After the loop, add the final segment (the rest of the transcript)
    final_segment = translated_transcript[last_split_point:].strip()
    # Add the final segment even if it's empty, to match segment count
    translated_segments.append(final_segment)


    # Final sanity check: Did we get the expected number of segments?
    if len(translated_segments) != num_english_segments:
        print(f"Error: Expected {num_english_segments} segments, but generated {len(translated_segments)}.")
        print("This might happen with empty segments or unexpected punctuation.")
        # Attempt to pad/truncate (less ideal)
        while len(translated_segments) < num_english_segments:
             translated_segments.append("")
        if len(translated_segments) > num_english_segments:
             translated_segments = translated_segments[:num_english_segments]
        # Return None because the process likely had issues
        return None

    return translated_segments

# --- Example Usage ---
# Using the same examples as before
eng_transcript = "The magini happen across a short mulyscript that describes a seene between a person, and their a i assistant. Thesgriptd has what the person asks the a eye, but the aeyes respons has been tourn off. Suppose you also have this powerful magical machine the cantake any text and provide a sensible prediction of what word comes next. They could then a finished thescript by feeding ind what you have to the machine, seeing what it would predict to start"
eng_segments = [
    'The magini happen across a short mulyscript that describes a seene between a person,', # Ends in ,
    'and their a i assistant.', # Ends in .
    'Thesgriptd has what the person asks the a eye,', # Ends in ,
    'but the aeyes respons has been tourn off.', # Ends in .
    'Suppose you also have this powerful magical machine the cantake any text and provide a', # Ends in a (FAIL!)
    'sensible prediction of what word comes next.', # Ends in .
    'They could then a finished thescript by feeding ind what you have to the machine,', # Ends in ,
    'seeing what it would predict to start' # No punctuation end
]
fr_transcript = "Les magini se produisent à travers un court mulyscrit qui décrit une vue entre une personne, et leur assistante. Thesgriptd a ce que la personne demande à l'œil, mais la responsabilité des yeux a été tournée. Supposons que vous avez aussi cette machine magique puissante le cantake n'importe quel texte et fournir une prédiction raisonnable de ce mot vient ensuite. Ils pourraient alors un script fini en nourrissant ce que vous avez à la machine, voir ce qu'il pourrait prédire pour commencer"

print("--- Attempting Split by Punctuation ---")
translated_splits_punct = split_translated_transcript_by_punctuation(
    eng_segments, fr_transcript
)

if translated_splits_punct is not None:
    print(f"Successfully Split Translated Segments ({len(translated_splits_punct)}):")
    for i, seg in enumerate(translated_splits_punct):
        print(f"{i}: {seg}")
else:
    print("\nSplit by punctuation failed or produced inconsistent results.")
