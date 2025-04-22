# Pipeline de Sous-titrage Vidéo

Ce projet fournit une pipeline pour générer des sous-titres pour des vidéos, corriger les transcriptions, traduire les sous-titres en plusieurs langues et les intégrer à la vidéo. L'application est construite avec Python et Gradio pour l'interface utilisateur.

## Fonctionnalités

- Extraction audio à partir de fichiers vidéo.
- Prédiction de transcriptions à l'aide d'un modèle d'apprentissage automatique.
- Correction des transcriptions avec vérification orthographique ou un modèle de langage (LLM).
- Traduction des sous-titres en plusieurs langues (par exemple, anglais, français, espagnol).
- Ajout de sous-titres à la vidéo.

## Structure

Les fichiers pertinents pour l'application sont situés dans le dossier `PROPRE` :

- `src/` : Contient le code source de l'application.
  - `interface/` : Gère l'interface utilisateur basée sur Gradio.
  - `postprocessing/` : Contient les modules pour le traitement des transcriptions (par exemple, partitionnement, vérification orthographique).
  - `transcription/` : Gère la prédiction de transcription.
  - `translation/` : Gère la traduction des sous-titres.
  - `subtitles/` : Gère la création de fichiers de sous-titres et leur intégration.
- `data/` : Stocke les fichiers d'entrée et de sortie.
  - `input/` : Contient les fichiers vidéo et audio d'entrée.
  - `output/` : Stocke les sous-titres générés et les vidéos avec sous-titres.

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/MaelCrd/deep-asr-caption.git
   cd PROPRE
   ```

2. Configurez un environnement virtuel :

   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv-propre\Scripts\activate
   ```

3. Installez les dépendances :

   ```bash
   # See the requirements.txt for the specific commands to install the dependencies
   ```

## Utilisation

1. Démarrez l'application :

   ```bash
   python src/app.py
   ```

2. Ouvrez l'interface Gradio dans votre navigateur. Vous verrez les options suivantes :
   - **Charger une vidéo** : Chargez le fichier vidéo que vous souhaitez traiter.
   - **Options de traitement** :
     - Activer la vérification orthographique pour la correction de transcription.
     - Activer la correction de transcription basée sur un LLM.
   - **Options de traduction** :
     - Choisir le modèle de traduction (par exemple, Marian-MT).
     - Sélectionner les langues cibles pour les sous-titres (par exemple, anglais, français).

3. Suivez ces étapes dans l'interface :
   1. Sélectionnez les options de traitement et de traduction souhaitées.
   2. Chargez un fichier vidéo.

4. Téléchargez la vidéo traitée avec sous-titres depuis l'interface.

## Fichiers de Sortie

- **Sous-titres** : Les fichiers de sous-titres générés sont enregistrés dans `data/output/subtitles/`.
- **Vidéos** : Les vidéos avec sous-titres intégrés sont enregistrées dans `data/output/videos/`.

## Licence

Ce projet est sous licence MIT.

## Auteurs

- Maël C.R.
- Virgil P.

*Projet réalisé dans le cadre du cours d'Apprentissage profond à l'Université du Québec à Chicoutimi (UQAC).*
