import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, GRU, Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense, Activation, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib


# Fonction de jetonisation (identique à avant)
def tokenize(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# Architecture du modèle
def create_model3(vocab_inp_size, vocab_tar_size, embedding_dim, units, input_seq_len, output_seq_len):
    """
    Crée un modèle encodeur-décodeur pour la traduction, en utilisant l'architecture de encdec_model.

    Args:
        vocab_inp_size: Taille du vocabulaire d'entrée (anglais).
        vocab_tar_size: Taille du vocabulaire cible (français).
        embedding_dim: Dimension des vecteurs d'embedding
        units: Nombre d'unités GRU.
        input_seq_len: Longueur maximale des séquences d'entrée.
        output_seq_len: Longueur maximale des séquences cibles.

    Returns:
        Modèle Keras construit, mais non entraîné.
    """

    learning_rate = 0.001

    ########### ENCODEUR ###########
    encoder_input = Input(shape=(input_seq_len,), name="input_Encoder")

    embeddings = Embedding(input_dim=vocab_inp_size, output_dim=embedding_dim, name="Embedding_layer")(encoder_input)

    encoder_output = Bidirectional(GRU(units, return_sequences=False, activation="tanh"), name="BiGRU_encoder")(embeddings)

    ########### INTERMÉDIAIRE ###########
    sequenced_context_vector = RepeatVector(output_seq_len)(encoder_output)

    ########### DÉCODEUR ###########
    x = GRU(units, return_sequences=True, activation="tanh", name="GRU_layer1")(sequenced_context_vector)
    x = GRU(units, return_sequences=True, activation="tanh", name="GRU_layer2")(x)

    preds = TimeDistributed(Dense(vocab_tar_size, activation="softmax"), name="Dense_layer")(x)

    ########### MODÈLE GLOBAL ###########
    model = Model(inputs=encoder_input, outputs=preds, name='Encdec')

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])

    return model

# Chargement des tokenizers
inp_lang = joblib.load('src/translation/custom_model/english_tokenizer.pkl')
targ_lang = joblib.load('src/translation/custom_model/french_tokenizer.pkl')

# Récupération des tailles de vocabulaire
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

# Définition des paramètres
embedding_dim = 256
units = 256
input_seq_len = 50  # Longueur maximale des séquences d'entrée (à ajuster si nécessaire)
output_seq_len = 50  # Longueur maximale des séquences cibles (à ajuster si nécessaire)

# Création du modèle
model = create_model3(vocab_inp_size, vocab_tar_size, embedding_dim, units, input_seq_len, output_seq_len)

# Charger les poids du modèle depuis un checkpoint
model.load_weights("src/translation/custom_model/cp-0034.weights.h5")

def preprocess_english_sentences(english_sentences, inp_lang, max_length):
    """Prétraite les phrases anglaises pour le modèle.

    Args:
        english_sentences: Liste des phrases anglaises.
        inp_lang: Objet TextVectorization pour le vocabulaire anglais.
        max_length: Longueur maximale des séquences d'entrée.

    Returns:
        Tenseur TensorFlow contenant les séquences d'entrée prétraitées.
    """
    input_tensor = inp_lang.texts_to_sequences(english_sentences)
    input_tensor = pad_sequences(input_tensor, maxlen=max_length, padding='post')
    input_tensor = tf.convert_to_tensor(input_tensor)
    return input_tensor

def translate(english_sentences, model= model, inp_lang = inp_lang, targ_lang= targ_lang, max_length=output_seq_len, progress_callback=None):
    """Traduit les phrases anglaises en français.

    Args:
        english_sentences: Liste des phrases anglaises.
        model: Modèle de traduction.
        inp_lang: Objet TextVectorization pour le vocabulaire anglais.
        targ_lang: Objet TextVectorization pour le vocabulaire français.
        max_length: Longueur maximale des séquences cibles.

    Returns:
        Liste des phrases françaises traduites.
    """
    input_tensor = preprocess_english_sentences(english_sentences, inp_lang, max_length)
    predictions = model.predict(input_tensor)
    predicted_sentences = []
    for prediction in predictions:
        predicted_sentence = []
        for word_index in prediction:
            predicted_word = targ_lang.index_word.get(tf.argmax(word_index).numpy())
            if predicted_word is None:
                break
            predicted_sentence.append(predicted_word)
        predicted_sentences.append(' '.join(predicted_sentence))
    return predicted_sentences

# # Exemple de phrases anglaises
# english_sentences = [
#     "Hello, how are you?",
#     "What is your name?",
#     "I like to play soccer.",
#     "Hello",
#     "Goodbye",
#     "The car is red.",
#     "I am a student.",
#     "The sky is blue.",
#     "I am learning how to code deep learning models with Python.",
#     "Traditionally, the language was thought to be closer to English than other languages in the West.",
#     "The French language is derived from the Latin language of the Roman Empire."

# ]

# # Traduction des phrases anglaises
# translated_sentences = translate(english_sentences, model, inp_lang, targ_lang, output_seq_len)

# # Affichage des résultats
# for english_sentence, translated_sentence in zip(english_sentences, translated_sentences):
#     print(f"English: {english_sentence}")
#     print(f"French: {translated_sentence}")
#     print("-" * 20)
