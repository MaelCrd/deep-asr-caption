import re
from itertools import groupby
import time
import numpy as np


def group_and_timestamps(input_pred):
    """
    Group the words in the input prediction and calculate the normalized start indices of each word.\n
    Ex:\n
    Input :\n
        input_pred = "----------------------s-o----mee--  -a-------s--s--i--d----  -i----ss--- --o---f--tt-enn--"\n
    Outputs :\n
        final_words = ['some', 'asid', 'is', 'often']\n
        start_indices = [0.0245, 0.467879, 0.6867, 0.977]
    """
    
    words_grouped = []
    start_indices = []
    current_index = 0

    for char in input_pred:
        if char.isalpha():
            if not words_grouped or not words_grouped[-1]:
                start_indices.append(current_index)
                words_grouped.append("")
            words_grouped[-1] += char
        elif words_grouped and words_grouped[-1]:
            if char == "-":
                # words_grouped[-1] += char
                pass
            elif char == "," or char == ".":
                words_grouped[-1] += char
            else:
                words_grouped.append("") # Start a new potential word
        current_index += 1

    # Remove any empty strings that might have been added
    words_grouped = [word for word in words_grouped if word]
    
    final_words = []
    for word in words_grouped:
        final_words.append("".join(list(g[0] for g in groupby(word))))
    # print("\nFinal words:")
    # print(" ".join(final_words))
    
    # Normalize the start indices
    start_indices = np.array(start_indices) / len(input_pred)
    # print("\nStart indices:")
    # print(start_indices)
    
    return final_words, start_indices


if __name__ == "__main__":
    input_string = "----------------------s-o----mee--  -a-------s--s--i--d----  -i----ss--- --o---f--tt-enn-- --a----d--d-e-d--  two-----------  -i-nn----hhi-----b-b-i--t-------  c--ll-i-mm-err--i-s---a----t---iio--n------  -i---n---  the-  two-----------be----------------------------------------------------.  -i-f-- yyou--  w-a-nnt  -a----  ssp-e-eed  u---p---- the- ss-e-t-t-ing  of- ss-i---p-er--l--uu-------------------, --o---nee  w-ay-----  -i-s--  t-o- hha----dd-  m-o--rre-  -n-eg-ggedt-of  -i---------o-----n-----ss-----, the---------  -inn---i-t----ii----a---t-orrss--------------- tha-t-------  ssttaa-rr--t--- the----- p----ll-e-mmerr--i--s---a--t---iio-n---------- rr-e----a----c--t--iionn-----------------------.  y-ou--- c-ann---------  b-y-----------  -ex---c-e---lllerrraa--t-orr--ss----  sp-e----s-i---f-i-cal-lly-- f-or- thiss  -b-ur---p-use- -of-f-  the  shell-------f-----------------------------."
    
    sentence, timestamps = group_and_timestamps(input_string)
    
    print("\nSentence:")
    print(sentence)
    print("\nTimestamps:")
    print(timestamps)