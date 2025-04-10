from transformers import MarianMTModel, MarianTokenizer

# Dictionnaire pour stocker les modèles et tokenizers chargés
modeles_tokenizers = {}

def charger_modele_tokenizer(langue_cible):
    """Charge le modèle et le tokenizer MarianMT pour la langue cible."""
    if langue_cible == 'fr':
        model_name = 'Helsinki-NLP/opus-mt-en-fr'
    elif langue_cible == 'es':
        model_name = 'Helsinki-NLP/opus-mt-en-es'
    else:
        raise ValueError(f"Langue cible '{langue_cible}' non supportée. Les options sont 'fr' ou 'es'.")

    if model_name not in modeles_tokenizers:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        modeles_tokenizers[model_name] = (model, tokenizer)

    return modeles_tokenizers[model_name]

# Pré-chargement du modèle et du tokenizer pour le français
charger_modele_tokenizer('fr')
charger_modele_tokenizer('es')


def traduire_liste(phrases_anglaises, output_language='fr', progress_callback=None):
    """Traduit une liste de phrases anglaises vers la langue cible spécifiée."""
    model, tokenizer = charger_modele_tokenizer(output_language)
    phrases_traduites = []
    for phrase_anglaise in phrases_anglaises:
        input_ids = tokenizer.encode(phrase_anglaise, return_tensors="pt")
        outputs = model.generate(input_ids)
        phrase_traduit = tokenizer.decode(outputs[0], skip_special_tokens=True)
        phrases_traduites.append(phrase_traduit)
        if progress_callback:
            progress_callback(len(phrases_traduites) / len(phrases_anglaises))
    return phrases_traduites

if __name__ == "__main__":
    liste_phrases_anglaises = [
        "Hello, how are you today?",
        "The weather is beautiful in Saguenay.",
        "I enjoy learning new things.",
        "This is a test sentence.",
        'The magini happen across a short mulyscript that describes a seene between a person, and their a i assistant. Thesgriptd has what the person asks the a eye, but the aeyes respons has been tourn off. Suppose you also have this powerful magical machine the cantake any text and provide a sensible prediction of what word comes next. They could then a finished thescript by feeding ind what you have to the machine, seeing what it would predict to start'
    ]

    # Traduction vers le français
    phrases_francaises = traduire_liste(liste_phrases_anglaises, output_language='fr')
    print("Traductions vers le français:")
    for original, traduit in zip(liste_phrases_anglaises, phrases_francaises):
        print(f"Anglais: {original}")
        print(f"Français: {traduit}")

    # print("\n" + "="*30 + "\n")

    # # Traduction vers l'espagnol
    # phrases_espagnoles = traduire_liste(liste_phrases_anglaises, output_language='es')
    # print("Traductions vers l'espagnol:")
    # for original, traduit in zip(liste_phrases_anglaises, phrases_espagnoles):
    #     print(f"Anglais: {original}")
    #     print(f"Espagnol: {traduit}")