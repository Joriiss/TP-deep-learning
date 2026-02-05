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

Analyse de **sentiment** sur des avis : Vader (NLTK), modèle transformer (Hugging Face) et Naive Bayes personnalisé (TF-IDF + MultinomialNB) entraîné sur des textes positifs/négatifs.

- **model_sentiment.py** — Entraîne le pipeline TF-IDF + Naive Bayes sur `TrainingDataPositive.txt` et `TrainingDataNegative.txt`, sauvegarde `modele_custom_nb.joblib`.
- **sentiment.py** — Lit `TestReviews.csv`, applique les trois méthodes (Vader, transformer, Naive Bayes) et écrit les résultats dans `sentiment.csv`.

Voir le `README.md` dans le dossier pour l’installation et les commandes.

## tp-classifications

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
