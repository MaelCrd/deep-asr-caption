from spellchecker import SpellChecker

SPELL = SpellChecker(distance=1)


def correct_sentence(sentence: str) -> str:
    """
    Correct the given sentence using a basic spell-checking on 1-letter change when only 1 candidate (safe).\n
    Ex:\n
        Note: sentence can be a string or a list of strings.\n
        sentence = "some asid is often aded two inhibit climerisation in the twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."\n
    Output:\n
        "Some acid is often added to inhibit climerisation in the tube. If you want to speed up the setting of siperlu, one way is to add more negedtof ions, the initiators that start the polymerisation reaction. You can buy accelerators specifically for this purpose off the shelf."
    """
    
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    elif not isinstance(sentence, str):
        raise ValueError("Input sentence must be a string or a list of strings.")
    
    # Words
    words = sentence.split()
    new_words = []
    for word_dirty in words:
        word = word_dirty.replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip()
        
        candidates = SPELL.candidates(word)
        corr = word  # Default to the original word
        if candidates is not None:
            if len(candidates) == 1:
                # If there's only one candidate, it is likely the correct spelling
                temp_corr = candidates.pop()
                if temp_corr != word:
                    print(word, '>', temp_corr)
                    corr = temp_corr
        
        # If '.' or ',' or ... is in the word, add it back to the corrected word
        if '.' in word_dirty:
            corr += '.'
        if ',' in word_dirty:
            corr += ','
        if '?' in word_dirty:
            corr += '?'
        if '!' in word_dirty:
            corr += '!'

        new_words.append(corr if corr != word else word_dirty)
    
    return " ".join(new_words)


if __name__ == "__main__":
    sentence = "some assid is often added two inhibbit climerisation in the twobe. if you want a speed up the setting of siperlu, one way is to had more neggedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesifically for this burpuse off the shelf."
    corrected = correct_sentence(sentence)
    print("Input sentence:")
    print(sentence)
    print("\nCorrected sentence:")
    print(corrected)