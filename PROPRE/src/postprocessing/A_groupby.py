from itertools import groupby
import numpy as np


def group_and_timestamps(input_pred: str) -> tuple[list[str], list[int]]:
    """
    Group the words in the input prediction and calculate the non-normalized start indices of each word.\n
    Ex:\n
    Input :\n
        input_pred = "----------------------s-o----mee--  -a-------s--s--i--d----  -i----ss--- --o---f--tt-enn--"\n
    Outputs :\n
        sentence = "some assid is often"
        start_indices = [23, 46, 52, 89]
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
                words_grouped[-1] += char # Comment to prevent too many letters in a row but seems best to keep it
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
    
    # # Normalize the start indices
    # start_indices = np.array(start_indices) / len(input_pred)
    # # print("\nStart indices:")
    # # print(start_indices)
    
    sentence = " ".join(final_words).replace("-", "").replace("  ", " ").strip()
    return sentence, start_indices


if __name__ == "__main__":
    input_string = "----------------------s-o----mee--  -a-------s--s--i--d----  -i----ss--- --o---f--tt-enn-- --a----d--d-e-d--  two-----------  -i-nn----hhi-----b-b-i--t-------  c--ll-i-mm-err--i-s---a----t---iio--n------  -i---n---  the-  two-----------be----------------------------------------------------.  -i-f-- yyou--  w-a-nnt  -a----  ssp-e-eed  u---p---- the- ss-e-t-t-ing  of- ss-i---p-er--l--uu-------------------, --o---nee  w-ay-----  -i-s--  t-o- hha----dd-  m-o--rre-  -n-eg-ggedt-of  -i---------o-----n-----ss-----, the---------  -inn---i-t----ii----a---t-orrss--------------- tha-t-------  ssttaa-rr--t--- the----- p----ll-e-mmerr--i--s---a--t---iio-n---------- rr-e----a----c--t--iionn-----------------------.  y-ou--- c-ann---------  b-y-----------  -ex---c-e---lllerrraa--t-orr--ss----  sp-e----s-i---f-i-cal-lly-- f-or- thiss  -b-ur---p-use- -of-f-  the  shell-------f-----------------------------."
    # input_string = "---------------------------------------------------------------------the- mma--g---i-nn--i---  hha-p--p-en-  a--crr-o---s-ss-- --a  ssho--rr--t---  m-u----l-y-----sscrri---p-tt----------- that- d-e--ssscrri----bess  -a- -s-e----e-ne- b-e---ttw-e---en---  -a-  p-e-rr--s----o---n-----------------,  -a-ndd  thei-rr--- --a----  -i----- -a-s---s-i--s---tt-a--n--t----------------. the---s-gri-ppttd hha-------s--- wha--t-  the  p-errr-s----o-n----  -a-----s--kk--s--- the--  --a----  -e-------yyye------------------,  b-u-------t--------------- the--  -a-------e--yyess-- rre----sspp-o------n--ss-----  ha--s--  bbe-enn  t-o--u-rr-nn-  -of---------ff---------------------------------. s-u-p-p-o----sse  yoou   al-----s-o----  ha--------vee---  thi-ss----  p-ow------err--f-ull-- -m-a--g---i--c-al-  m-a---chi------mee---  the  c-an---t-a----k-e---  -a--n---y--  t-e-----x----tt--------------  -and  pro----v-i------dee-  --a- -s-i-n--s---i--b-lee-  pre------d-i-c-tt-iiio--n-----  -o-f-- wwha--t---  w-o-rr----d---  c-o--m-e-ss--  -n-e-----x-----tt------------------------. they-- could the--n-  a--- f-i-nn--ishhed- the---s-cri---p-tt---- b-y----  f-e--e--d-ingg-  i-nd  wha-t-  yyou-  hha------vee  t-o-  the m-a--chi---------nee------------------,  s-e--e--inng  wha-t- -it would-  pre-----d-i---cctt--------  t-o-------  ssst-a-rr--tt-"
    
    sentence, timestamps = group_and_timestamps(input_string)
    
    print("\nSentence:")
    print(sentence)
    print("\nTimestamps:")
    print(timestamps)