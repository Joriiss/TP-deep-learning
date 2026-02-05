# TP Python

Travaux pratiques Python — génération de formes et traduction en langage naturel.

## tp-formes

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

## tp-classifications-2

Même principe en **non supervisé** sur ~60 000 images de **voitures** dans `data/` (dossier plat, pas de sous-dossiers). Les images proviennent du jeu de données [The Car Connection Picture Dataset](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset/data) (Kaggle). Le **nombre de clusters k** est déduit automatiquement du nombre de marques distinctes : la marque est prise comme premier mot du nom de fichier (ex. `Acura_ILX_2013_...` → Acura).

**Contenu du dossier :**
- `train_clusters.py` — Script principal : charge les images, extrait les descripteurs par lots (ResNet18), lance K-means (k = nombre de marques), écrit les sorties dans `output/`.
- `requirements.txt` — Dépendances (torch, torchvision, Pillow, scikit-learn, numpy).
- `data/` — À remplir avec les images du dataset Kaggle (fichiers `.jpg` à la racine de `data/`).
- `output/` — Créé à l’exécution : `kmeans_model.pkl`, `features_cache.npz`, `cluster_assignments.csv` (path, cluster_id, brand), `missclassified.csv`. En console : effectifs par cluster, composition par marque, résumé des missclassés.

### Installation

```bash
cd tp-classifications-2
pip install -r requirements.txt
```

### Lancer

```bash
python train_clusters.py
```
