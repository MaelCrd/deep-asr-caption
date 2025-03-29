
a = "the magini happen across a short mulyscript that describes a seene between a person, and their a i assistant. thesgriptd has what the person asks the a eye, but the aeyes respons has been tourn off. suppose you also have this powerful magical machime the cantake any text and provide a sinsible prediction of what word comes next. they could then a finished thescript by feeding ind what you have to the machine, seeing what it would predict to start"

a = "some assid is often added two inhibbit climerisation in the twobe. if you want a speed up the setting of siperlu, one way is to had more neggedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesifically for this burpuse off the shelf."

from spellchecker import SpellChecker
spell = SpellChecker(distance=1)

# find those words that may be misspelled
words = a.split()
# misspelled = spell.unknown(words)
 
new_words = []
for word_dirty in words:
    word = word_dirty.replace('.', '').replace(',', '').strip()
    
    candidates = spell.candidates(word)
    corr = word  # Default to the original word
    if candidates is not None:
        if len(candidates) == 1:
            # If there's only one candidate, it is likely the correct spelling
            temp_corr = candidates.pop()
            if temp_corr != word:
                print(word, '>', temp_corr)
                corr = temp_corr
    
    # If '.' or ',' is in the word, add it back to the corrected word
    if '.' in word_dirty:
        corr += '.'
    if ',' in word_dirty:
        corr += ','
    
    new_words.append(corr if corr != word else word_dirty)
 
    # if word != spell.correction(word):
    #     # Get a list of `likely` options
    #     print(word, spell.candidates(word))

print("Original sentence:")
print(a)
print("\nCorrected sentence:")
print(" ".join(new_words))

# from spellchecker import SpellChecker
# spell = SpellChecker()

# def check_two_word_combination(word, spell_checker):
#     n = len(word)
#     for i in range(1, n):
#         part1 = word[:i]
#         part2 = word[i:]
#         if part1 in spell_checker and part2 in spell_checker:
#             return [part1, part2]
#     return None

# # Split the input string into words
# words = a.split()
# corrected_words = []

# for original_word in words:
#     cleaned_word = original_word.replace('.', '').replace(',', '').strip()
#     correction = spell.correction(cleaned_word)

#     if cleaned_word != correction:
#         corrected_words.append(correction if correction else cleaned_word)
#     else:
#         two_word_correction = check_two_word_combination(cleaned_word, spell)
#         if two_word_correction:
#             corrected_words.extend(two_word_correction)
#         else:
#             corrected_words.append(original_word) # Keep the original if no correction

# print(" ".join(corrected_words))


####################################################

# from spello.model import SpellCorrectionModel  
# sp = SpellCorrectionModel(language='en')

# sp.load('en_large.pkl')

# print(sp.spell_correct(a))