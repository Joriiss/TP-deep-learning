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
