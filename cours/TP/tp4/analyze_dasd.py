import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "my_data/results.json"
LOG_STAGE1 = "saves/qwen_dasd/stage1/trainer_log.jsonl"
LOG_STAGE2 = "saves/qwen_dasd/stage2_final/trainer_log.jsonl"


def plot_dataset_stats():
    print(f"Analyse du dataset : {DATA_FILE}...")

    if not os.path.exists(DATA_FILE):
        print(f"Fichier {DATA_FILE} introuvable.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    teacher_scores_s1 = [d["stage1"]["probability"] for d in data]
    teacher_scores_s2 = [d["stage2"]["probability"] for d in data]
    lengths_s1 = [len(d["stage1"]["answer"]) for d in data]
    lengths_s2 = [len(d["stage2"]["answer"]) for d in data]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].hist(teacher_scores_s1, bins=30, color="skyblue", edgecolor="black", alpha=0.7, label="Stage 1 (τ=0.3)")
    axs[0].hist(teacher_scores_s2, bins=30, color="salmon", edgecolor="black", alpha=0.5, label="Stage 2 (τ=0.9)")
    axs[0].set_title("Confiance du Teacher (Probabilité géométrique)")
    axs[0].set_xlabel("P_teacher")
    axs[0].set_ylabel("Nombre d'exemples")
    axs[0].legend()
    axs[0].grid(axis="y", alpha=0.3)

    axs[1].hist(lengths_s1, bins=30, color="skyblue", edgecolor="black", alpha=0.7, label="Stage 1 (τ=0.3)")
    axs[1].hist(lengths_s2, bins=30, color="salmon", edgecolor="black", alpha=0.5, label="Stage 2 (τ=0.9)")
    axs[1].set_title("Distribution des longueurs de réponses")
    axs[1].set_xlabel("Nombre de caractères")
    axs[1].set_ylabel("Nombre d'exemples")
    axs[1].legend()
    axs[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph_dataset_stats.png", dpi=150)
    print("Sauvegardé : graph_dataset_stats.png")

    print(f"  Nombre d'exemples    : {len(data)}")
    print(f"  Confiance S1 (moy)   : {np.mean(teacher_scores_s1):.4f}")
    print(f"  Confiance S2 (moy)   : {np.mean(teacher_scores_s2):.4f}")
    print(f"  Longueur S1 (moy)    : {np.mean(lengths_s1):.0f} chars")
    print(f"  Longueur S2 (moy)    : {np.mean(lengths_s2):.0f} chars")


def load_trainer_log(log_path):
    steps, losses = [], []
    if not os.path.exists(log_path):
        return steps, losses

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "loss" in entry and "current_steps" in entry:
                    steps.append(entry["current_steps"])
                    losses.append(entry["loss"])
                elif "loss" in entry and "step" in entry:
                    steps.append(entry["step"])
                    losses.append(entry["loss"])
            except Exception:
                continue
    return steps, losses


def plot_training_curves():
    print("\nAnalyse de l'entraînement...")

    s1_steps, s1_loss = load_trainer_log(LOG_STAGE1)
    s2_steps, s2_loss = load_trainer_log(LOG_STAGE2)

    if not s1_steps and not s2_steps:
        print("Aucun log d'entraînement trouvé.")
        return

    plt.figure(figsize=(10, 5))

    if s1_steps:
        plt.plot(s1_steps, s1_loss, label="Stage 1 (τ=0.3)", color="blue", linewidth=2)
        print(f"  Stage 1 : {len(s1_steps)} points de log, loss finale = {s1_loss[-1]:.4f}")

    if s2_steps:
        plt.plot(s2_steps, s2_loss, label="Stage 2 (τ=0.9)", color="red", linewidth=2, linestyle="--")
        print(f"  Stage 2 : {len(s2_steps)} points de log, loss finale = {s2_loss[-1]:.4f}")

    plt.title("Courbes d'apprentissage — Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graph_training_loss.png", dpi=150)
    print("Sauvegardé : graph_training_loss.png")


if __name__ == "__main__":
    plot_dataset_stats()
    plot_training_curves()
    print("\nTerminé.")
