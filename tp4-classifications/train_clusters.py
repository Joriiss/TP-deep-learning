"""
Unsupervised clustering of images in data/.
Loads all images without using folder names as labels, extracts features
with a pretrained CNN, then fits K-means to discover clusters.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


def get_image_paths(data_dir: Path) -> list[tuple[Path, str]]:
    """List all image paths and true label (folder name) for evaluation only."""
    pairs = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = subdir.name
        for p in subdir.glob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                pairs.append((p, label))
    return pairs


def load_features(paths_with_labels: list[tuple[Path, str]], device: torch.device) -> tuple[np.ndarray | None, list[str], list[tuple[Path, str]]]:
    """Load images and extract features with pretrained ResNet18 (no labels used for training).
    Returns (feature matrix, true labels, list of (path, label) for loaded images).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove last FC
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    features_list = []
    true_labels = []
    loaded_paths = []
    for path, label in paths_with_labels:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x)
        features_list.append(f.cpu().numpy().ravel())
        true_labels.append(label)
        loaded_paths.append((path, label))

    X = np.vstack(features_list) if features_list else None
    return X, true_labels, loaded_paths


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    if not data_dir.is_dir():
        print(f"Data folder not found: {data_dir}")
        return

    paths_with_labels = get_image_paths(data_dir)
    if not paths_with_labels:
        print("No images found in data/ subfolders.")
        return

    print(f"Found {len(paths_with_labels)} images (labels used only for evaluation).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features on {device}...")

    X, true_labels, loaded_paths = load_features(paths_with_labels, device)
    if X is None:
        print("Could not load any image.")
        return

    n_clusters = 3
    print(f"Fitting K-means with k={n_clusters} (unsupervised)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    model_path = out_dir / "kmeans_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Model saved to {model_path}")

    results_path = out_dir / "cluster_assignments.csv"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("path,cluster_id,true_label\n")
        for (p, label), cid in zip(loaded_paths, cluster_ids):
            f.write(f'"{p}",{cid},{label}\n')
    print(f"Assignments saved to {results_path}")

    # Optional: report cluster vs true label (model never saw labels)
    from collections import Counter
    print("\nCluster composition (true labels):")
    for c in range(n_clusters):
        mask = cluster_ids == c
        labels_in_c = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
        counts = Counter(labels_in_c)
        print(f"  Cluster {c}: {dict(counts)}")

    # Majority label per cluster (for "missclassification" vs true labels)
    cluster_to_majority = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        labels_in_c = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
        if labels_in_c:
            cluster_to_majority[c] = Counter(labels_in_c).most_common(1)[0][0]
        else:
            cluster_to_majority[c] = None

    # Missclassified: true label != majority label of assigned cluster
    missclassified = []
    for i, ((p, true_label), cid) in enumerate(zip(loaded_paths, cluster_ids)):
        majority = cluster_to_majority.get(cid)
        if majority is not None and true_label != majority:
            missclassified.append((p, cid, true_label, majority))

    miss_path = out_dir / "missclassified.csv"
    with open(miss_path, "w", encoding="utf-8") as f:
        f.write("path,cluster_id,true_label,majority_in_cluster\n")
        for p, cid, true_label, majority in missclassified:
            f.write(f'"{p}",{cid},{true_label},{majority}\n')
    print(f"\nMissclassified (true label ≠ majority of cluster): {len(missclassified)} images")
    print(f"List saved to {miss_path}")
    for p, cid, true_label, majority in missclassified:
        print(f"  {p.name}: true={true_label} → cluster {cid} (majority: {majority})")


if __name__ == "__main__":
    main()
