import json
import matplotlib.pyplot as plt
import numpy as np

# Charger les données
with open("analysis_dasd.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraire les scores
teacher_scores = [d["stats"]["teacher_mean_logprob"] for d in data]
lengths_filtered = [d["stats"]["length_filtered"] for d in data]

# 1. Histogramme de la confiance du Teacher
plt.figure(figsize=(10, 5))
plt.hist(teacher_scores, bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution de la Confiance du Teacher (Logprobs)")
plt.xlabel("Logprob Moyen (Proche de 0 = Très confiant)")
plt.ylabel("Nombre d'exemples")
plt.grid(axis='y', alpha=0.75)
plt.savefig("histogram_teacher_confidence.png")
print("✅ Graphique sauvegardé : histogram_teacher_confidence.png")

# 2. Stats textuelles pour le rapport
print(f"\n--- STATISTIQUES DASD ---")
print(f"Nombre d'exemples : {len(data)}")
print(f"Confiance Moyenne Teacher : {np.mean(teacher_scores):.4f}")
print(f"Longueur moyenne des réponses : {np.mean(lengths_filtered):.0f} caractères")