# TP Python

Travaux pratiques Python — génération de formes et traduction en langage naturel.

## TP1 Génération de formes géométriques

Deux scripts pour créer des images avec des formes (triangles, carrés, cercles, rectangles, étoiles) :

- **shapes_generator.py** — Saisie manuelle : vous entrez la forme, la couleur (hex), la taille et la position. Vous pouvez ajouter plusieurs formes. L’image finale est générée en 800×600 px.

- **translator.py** — Saisie en langage naturel : vous décrivez les formes (ex. *« un grand carré rouge en haut à droite et un petit cercle bleu au centre »*). Le script utilise l’API Gemini (clé dans `.env`) pour traduire en instructions, puis génère l’image. Chaque exécution enregistre dans le dossier `results/` une image et un fichier JSON (prompt + traduction en formes) avec un horodatage unique, sans écraser les anciens résultats.

### Installation

```bash
cd tp-formes
pip install -r requirements.txt
```

### Lancer

- Générateur manuel : `python shapes_generator.py`
- Traducteur (langage naturel) : `python translator.py`

La clé API Gemini doit être définie dans le fichier `.env` à la racine du projet (`GEMINI_API_KEY=...`).

## TP2 - Analyse de sentiment : Positif ou négatif

Analyse de **sentiment** sur des avis : trois méthodes sont utilisées — **Vader** (NLTK), un **modèle transformer** (Hugging Face) et un **Naive Bayes personnalisé** (TF-IDF + MultinomialNB) entraîné sur des textes positifs et négatifs.

- **model_sentiment.py** — Entraîne le pipeline TF-IDF + Naive Bayes sur les fichiers d’entraînement, sauvegarde le modèle dans `modele_custom_nb.joblib`. À lancer une fois avant d’utiliser le classifieur personnalisé dans `sentiment.py`.
- **sentiment.py** — Charge `TestReviews.csv`, applique les trois classifieurs (Vader, transformer, Naive Bayes) à chaque avis et enregistre les résultats dans `sentiment.csv` (colonnes `sentiment`, `sentiment_transformer`, `sentiment_nb_custom`).

**Données :** `TrainingDataPositive.txt` et `TrainingDataNegative.txt` (un avis par ligne) pour l’entraînement du Naive Bayes ; `TestReviews.csv` (colonne `review`) pour les prédictions. Résultat : `sentiment.csv`.

### Installation

```bash
cd tp2-sentiments
pip install pandas nltk scikit-learn transformers joblib
```

Au premier run, télécharger le lexique Vader :  
`python -c "import nltk; nltk.download('vader_lexicon')"`

### Lancer

1. Entraîner le modèle personnalisé (une fois) : `python model_sentiment.py`
2. Lancer l’analyse sur les avis de test : `python sentiment.py`

## TP4 - Classification d’images (clustering)

Apprentissage **non supervisé** sur les images du dossier `data/` (sous-dossiers : airplanes, Motorbikes, schooner). Les images proviennent du jeu de données [Caltech101 - Airplanes, Motorbikes, Schooners](https://www.kaggle.com/datasets/maricinnamon/caltech101-airplanes-motorbikes-schooners) (Kaggle). Le modèle ne reçoit pas les étiquettes : il apprend uniquement à former des clusters.

- **train_clusters.py** — Charge toutes les images, extrait des descripteurs avec un ResNet18 pré-entraîné, puis applique un K-means (k=3). Sauvegarde le modèle dans `output/kmeans_model.pkl` et les affectations dans `output/cluster_assignments.csv`. En fin d’exécution, affiche la composition de chaque cluster selon les vrais labels (pour analyse uniquement).

### Installation

```bash
cd tp-classifications
pip install -r requirements.txt
```

### Lancer

```bash
python train_clusters.py
```
