"""
Génération de datasets avec différentes distributions pour reverse-engineering.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(
    n_samples: int,
    weights: dict,
    force_penalty_threshold: float = None,  # Si force > threshold, pénalité
    force_penalty: float = 0.0,
    noise: float = 0.1,
    normalize: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère un dataset avec des poids personnalisés.

    weights: dict avec clés 'force', 'intelligence', 'agilite', 'chance',
             'experience', 'niveau_quete', 'equipement', 'fatigue'
    """
    np.random.seed(seed)

    # Générer les features
    data = {
        'force': np.random.uniform(0, 100, n_samples),
        'intelligence': np.random.uniform(0, 100, n_samples),
        'agilite': np.random.uniform(0, 100, n_samples),
        'chance': np.random.uniform(0, 100, n_samples),
        'experience': np.random.uniform(0, 20, n_samples),
        'niveau_quete': np.random.uniform(1, 10, n_samples),
        'equipement': np.random.uniform(0, 100, n_samples),
        'fatigue': np.random.uniform(0, 100, n_samples),
    }

    df = pd.DataFrame(data)

    # Normaliser pour le calcul du score
    df_norm = df.copy()
    for col in df.columns:
        df_norm[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    # Calculer le score de survie
    score = np.zeros(n_samples)

    # Facteurs positifs
    score += weights.get('force', 0) * df_norm['force']
    score += weights.get('intelligence', 0) * df_norm['intelligence']
    score += weights.get('agilite', 0) * df_norm['agilite']
    score += weights.get('chance', 0) * df_norm['chance']
    score += weights.get('experience', 0) * df_norm['experience']
    score += weights.get('equipement', 0) * df_norm['equipement']

    # Facteurs négatifs
    score -= weights.get('fatigue', 0) * df_norm['fatigue']
    score -= weights.get('niveau_quete', 0) * df_norm['niveau_quete']

    # Pénalité force élevée (arrogance)
    if force_penalty_threshold is not None:
        arrogant = df['force'] > force_penalty_threshold
        score[arrogant] -= force_penalty

    # Ajouter du bruit
    score += np.random.normal(0, noise, n_samples)

    # Convertir en probabilité puis en label
    prob = 1 / (1 + np.exp(-score))
    df['survie'] = (prob > 0.5).astype(int)

    if normalize:
        for col in df.columns[:-1]:  # Tout sauf 'survie'
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    return df


# === CONFIGURATIONS À TESTER ===

CONFIGS = {
    # Config originale (Terres Connues)
    "original": {
        "weights": {
            "equipement": 0.25,
            "force": 0.25,
            "intelligence": 0.20,
            "experience": 0.15,
            "agilite": 0.10,
            "chance": 0.05,
            "fatigue": 0.15,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": None,
        "force_penalty": 0,
    },

    # Terres Maudites (selon le README)
    "maudites_v1": {
        "weights": {
            "intelligence": 0.30,
            "agilite": 0.20,
            "chance": 0.20,
            "equipement": 0.15,
            "force": 0.10,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.15,
    },

    # Intelligence dominante
    "intel_heavy": {
        "weights": {
            "intelligence": 0.40,
            "agilite": 0.15,
            "chance": 0.15,
            "equipement": 0.10,
            "force": 0.05,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.20,
    },

    # Chance dominante
    "chance_heavy": {
        "weights": {
            "chance": 0.35,
            "intelligence": 0.20,
            "agilite": 0.15,
            "equipement": 0.10,
            "force": 0.05,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 60,
        "force_penalty": 0.15,
    },

    # Agilité dominante
    "agility_heavy": {
        "weights": {
            "agilite": 0.35,
            "intelligence": 0.20,
            "chance": 0.15,
            "equipement": 0.10,
            "force": 0.05,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.15,
    },

    # Force très pénalisée
    "force_punished": {
        "weights": {
            "intelligence": 0.25,
            "agilite": 0.20,
            "chance": 0.20,
            "equipement": 0.15,
            "force": 0.0,  # Force inutile
            "experience": 0.10,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 50,
        "force_penalty": 0.25,
    },

    # Mix équilibré sans force
    "balanced_no_force": {
        "weights": {
            "intelligence": 0.25,
            "agilite": 0.25,
            "chance": 0.20,
            "equipement": 0.15,
            "force": 0.0,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },
}


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Génération des datasets...")
    print("=" * 50)

    for name, config in CONFIGS.items():
        # Train set
        train_df = generate_dataset(
            n_samples=800,
            weights=config["weights"],
            force_penalty_threshold=config["force_penalty_threshold"],
            force_penalty=config["force_penalty"],
            noise=0.3,
            normalize=False,
            seed=42
        )

        # Val set
        val_df = generate_dataset(
            n_samples=200,
            weights=config["weights"],
            force_penalty_threshold=config["force_penalty_threshold"],
            force_penalty=config["force_penalty"],
            noise=0.3,
            normalize=False,
            seed=123
        )

        train_path = data_dir / f"train_{name}.csv"
        val_path = data_dir / f"val_{name}.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        survie_rate = train_df['survie'].mean()
        print(f"{name:20s}: train={train_path.name}, survie={survie_rate:.1%}")

    print("=" * 50)
    print("Datasets générés dans", data_dir)
    print("\nPour entraîner sur un dataset spécifique, modifie train_oracle.py")
    print("ou utilise le script train_custom.py")


if __name__ == "__main__":
    main()
