"""
Génération de datasets V2 - Fine-tuning basé sur les résultats.
Meilleurs: maudites_v1, balanced_no_force
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_dataset(
    n_samples: int,
    weights: dict,
    force_penalty_threshold: float = None,
    force_penalty: float = 0.0,
    noise: float = 0.1,
    normalize: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)

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

    df_norm = df.copy()
    for col in df.columns:
        df_norm[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    score = np.zeros(n_samples)

    score += weights.get('force', 0) * df_norm['force']
    score += weights.get('intelligence', 0) * df_norm['intelligence']
    score += weights.get('agilite', 0) * df_norm['agilite']
    score += weights.get('chance', 0) * df_norm['chance']
    score += weights.get('experience', 0) * df_norm['experience']
    score += weights.get('equipement', 0) * df_norm['equipement']

    score -= weights.get('fatigue', 0) * df_norm['fatigue']
    score -= weights.get('niveau_quete', 0) * df_norm['niveau_quete']

    if force_penalty_threshold is not None:
        arrogant = df['force'] > force_penalty_threshold
        score[arrogant] -= force_penalty

    score += np.random.normal(0, noise, n_samples)

    prob = 1 / (1 + np.exp(-score))
    df['survie'] = (prob > 0.5).astype(int)

    if normalize:
        for col in df.columns[:-1]:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    return df


# === CONFIGURATIONS V2 - Fine-tuning ===

CONFIGS = {
    # Variation 1: Mix des 2 meilleurs
    "mix_best": {
        "weights": {
            "intelligence": 0.28,  # Entre 0.25 et 0.30
            "agilite": 0.23,       # Entre 0.20 et 0.25
            "chance": 0.20,
            "equipement": 0.15,
            "force": 0.05,         # Très bas
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.10,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.12,
    },

    # Variation 2: Intelligence encore plus haute
    "intel_boost": {
        "weights": {
            "intelligence": 0.35,
            "agilite": 0.22,
            "chance": 0.18,
            "equipement": 0.12,
            "force": 0.03,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },

    # Variation 3: Agilité plus haute
    "agil_boost": {
        "weights": {
            "intelligence": 0.25,
            "agilite": 0.30,
            "chance": 0.18,
            "equipement": 0.12,
            "force": 0.05,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },

    # Variation 4: Force à zéro, pas de pénalité
    "zero_force": {
        "weights": {
            "intelligence": 0.28,
            "agilite": 0.25,
            "chance": 0.22,
            "equipement": 0.15,
            "force": 0.0,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": None,
        "force_penalty": 0,
    },

    # Variation 5: Équipement plus important
    "equip_boost": {
        "weights": {
            "intelligence": 0.25,
            "agilite": 0.22,
            "chance": 0.18,
            "equipement": 0.22,
            "force": 0.03,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },

    # Variation 6: Chance boost
    "chance_boost": {
        "weights": {
            "intelligence": 0.25,
            "agilite": 0.22,
            "chance": 0.28,
            "equipement": 0.12,
            "force": 0.03,
            "experience": 0.05,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },

    # Variation 7: Fatigue très pénalisante
    "fatigue_heavy": {
        "weights": {
            "intelligence": 0.28,
            "agilite": 0.23,
            "chance": 0.20,
            "equipement": 0.15,
            "force": 0.05,
            "experience": 0.05,
            "fatigue": 0.20,  # Plus pénalisant
            "niveau_quete": 0.12,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.12,
    },

    # Variation 8: Expérience inutile
    "no_exp": {
        "weights": {
            "intelligence": 0.30,
            "agilite": 0.25,
            "chance": 0.22,
            "equipement": 0.15,
            "force": 0.03,
            "experience": 0.0,
            "fatigue": 0.10,
            "niveau_quete": 0.08,
        },
        "force_penalty_threshold": 70,
        "force_penalty": 0.10,
    },
}


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Génération des datasets V2...")
    print("=" * 50)

    for name, config in CONFIGS.items():
        train_df = generate_dataset(
            n_samples=800,
            weights=config["weights"],
            force_penalty_threshold=config["force_penalty_threshold"],
            force_penalty=config["force_penalty"],
            noise=0.3,
            normalize=False,
            seed=42
        )

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
        print(f"{name:20s}: survie={survie_rate:.1%}")

    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
