"""
Script pour entraîner sur des datasets custom.
Usage: uv run train_custom.py --dataset maudites_v1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Nom du dataset (ex: maudites_v1, intel_heavy, ...)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Nombre de runs pour moyenner')
    args, remaining = parser.parse_known_args()

    data_dir = Path(__file__).parent / "data"
    train_path = data_dir / f"train_{args.dataset}.csv"
    val_path = data_dir / f"val_{args.dataset}.csv"

    if not train_path.exists():
        print(f"Dataset non trouvé: {train_path}")
        print("\nDatasets disponibles:")
        for f in data_dir.glob("train_*.csv"):
            name = f.stem.replace("train_", "")
            if name not in ["", "dungeon"]:
                print(f"  - {name}")
        return

    # Modifier temporairement les chemins dans train_oracle.py
    # Plus simple: on copie les fichiers vers train.csv et val.csv
    import shutil

    original_train = data_dir / "train.csv"
    original_val = data_dir / "val.csv"

    # Backup
    backup_train = data_dir / "train_backup.csv"
    backup_val = data_dir / "val_backup.csv"

    if original_train.exists() and not backup_train.exists():
        shutil.copy(original_train, backup_train)
    if original_val.exists() and not backup_val.exists():
        shutil.copy(original_val, backup_val)

    # Copier le dataset custom
    shutil.copy(train_path, original_train)
    shutil.copy(val_path, original_val)

    print(f"Dataset: {args.dataset}")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print("=" * 50)

    # Lancer l'entraînement
    cmd = [sys.executable, "train_oracle.py"] + remaining
    subprocess.run(cmd)

    # Restaurer
    if backup_train.exists():
        shutil.copy(backup_train, original_train)
    if backup_val.exists():
        shutil.copy(backup_val, original_val)


if __name__ == "__main__":
    main()
