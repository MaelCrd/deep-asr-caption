import difflib


def correlate_sequences_with_timestamps(inital_sequence: str, initial_timestamps: list[int], corrected_sentence: str) -> list[int]:
    """
    Correlate the initial sequence of words with timestamps to a corrected sentence.\n
    Ex:\n
    Input :\n
        inital_sequence = "some asid is often aded two inhibit climerisation in the he twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."\n
        initial_timestamps = [22, 37, 62, 75, 93, 110, 127, 160, 206, 216, 219, 222, 294, 300, 308, 318, 325, 336, 346, 351, 365, 369, 410, 419, 431, 438, 443, 455, 467, 483, 514, 529, 575, 589, 605, 614, 661, 711, 719, 735, 752, 789, 817, 823, 831, 846, 853, 858]\n
        corrected_sentence = "Some acid is often added to inhibit polymerization in the tube. If you want to speed up the setting of superglue, one way is to add more negative ions, the initiators that start the polymerization reaction. You can buy accelerators specifically for this purpose off the shelf."\n
    Outputs :\n
        Adjusted timestamps: \n
            [22, 37, 62, 75, 93, 110, 127, 160, 206, 216, 219, 294, 300, 308, 318, 325, 336, 346, 351, 365, 369, 410, 419, 431, 438, 443, 455, 467, 483, 514, 529, 575, 589, 605, 614, 661, 711, 719, 735, 752, 789, 817, 823, 831, 846, 853, 858]
    """
    
    inital_sequence = inital_sequence.lower()
    corrected_sentence = corrected_sentence.lower()

    # predicted_words = inital_sequence.split()
    # corrected_words = corrected_sentence.split()

    predicted_words_no_punct = [word.lower() for word in inital_sequence.split() if word not in ['.', ',']]
    corrected_words_no_punct = [word.lower() for word in corrected_sentence.split() if word not in ['.', ',']]

    matcher = difflib.SequenceMatcher(None, predicted_words_no_punct, corrected_words_no_punct)

    predicted_timestamps_extended = initial_timestamps + [None] * (len(predicted_words_no_punct) - len(initial_timestamps))

    adjusted_timestamps = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == 'equal' or op == 'replace':
            for i in range(b0, b1):
                if a0 < len(predicted_timestamps_extended):
                    adjusted_timestamps.append(predicted_timestamps_extended[a0])
                elif adjusted_timestamps:
                    adjusted_timestamps.append(adjusted_timestamps[-1])
                else:
                    adjusted_timestamps.append(0)
                a0 += 1
        elif op == 'insert':
            for i in range(b0, b1):
                if adjusted_timestamps:
                    adjusted_timestamps.append(adjusted_timestamps[-1])
                else:
                    adjusted_timestamps.append(0)
        elif op == 'delete':
            a0 += (a1 - a0)

    # print(f"Number of corrected words: {len(corrected_words_no_punct)}")
    # print(f"Number of adjusted timestamps: {len(adjusted_timestamps)}")
    # print(f"Inital timestamps: \n{predicted_timestamps_extended}")
    # print(f"Adjusted timestamps: \n{adjusted_timestamps}")

    # # Print the original and corrected text with timestamps
    # for i in range(max(max(adjusted_timestamps), max(initial_timestamps)) + 1):
    #     printed = False
    #     if i in initial_timestamps:
    #         print(i, 'PRED', end=' ')
    #         print(predicted_words_no_punct[initial_timestamps.index(i)], end=' ')
    #         print()
    #         printed = True
    #     if i in adjusted_timestamps:
    #         print(i, 'AJUS', end=' ')
    #         print(corrected_words_no_punct[adjusted_timestamps.index(i)], end=' ')
    #         print()
    #         printed = True
    #     if printed:
    #         print()
    #         input("Press Enter to continue...")
    
    return adjusted_timestamps

    
if __name__ == "__main__":
    _predicted = "some asid is often aded two inhibit climerisation in the he twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."
    _predicted_timestamps = [22, 37, 62, 75, 93, 110, 127, 160, 206, 216, 219, 222, 294, 300, 308, 318, 325, 336, 346, 351, 365, 369, 410, 419, 431, 438, 443, 455, 467, 483, 514, 529, 575, 589, 605, 614, 661, 711, 719, 735, 752, 789, 817, 823, 831, 846, 853, 858]
    _corrected = "Some acid is often added to inhibit polymerization in the tube. If you want to speed up the setting of superglue, one way is to add more negative ions, the initiators that start the polymerization reaction. You can buy accelerators specifically for this purpose off the shelf."
    
    adjusted_timestamps = correlate_sequences_with_timestamps(_predicted, _predicted_timestamps, _corrected)
    
    print("\nAdjusted timestamps:")
    print(adjusted_timestamps)
    