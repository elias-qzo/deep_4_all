import json

# 1. Mets le nom du fichier de ton collègue ici
INPUT_FILE = "my_data/results.json" 

# Fichiers de sortie pour Llama-Factory
OUTPUT_STAGE1 = "my_data/stage1_low_temp.json"
OUTPUT_STAGE2 = "my_data/stage2_high_temp.json"

def convert_dataset():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {INPUT_FILE} est introuvable.")
        return

    data_s1 = []
    data_s2 = []

    print(f"🔄 Conversion de {len(data)} entrées...")

    for entry in data:
        question = entry["question"]
        
        # --- Extraction Stage 1 (Low Temp) ---
        if "stage1" in entry and entry["stage1"]["passed_quality"]:
            answer_s1 = entry["stage1"]["answer"]
            data_s1.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer_s1}
                ]
            })

        # --- Extraction Stage 2 (High Temp) ---
        if "stage2" in entry and entry["stage2"]["passed_quality"]:
            answer_s2 = entry["stage2"]["answer"]
            data_s2.append({
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer_s2}
                ]
            })

    # Sauvegarde
    with open(OUTPUT_STAGE1, "w", encoding="utf-8") as f:
        json.dump(data_s1, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_STAGE2, "w", encoding="utf-8") as f:
        json.dump(data_s2, f, indent=2, ensure_ascii=False)

    print(f"✅ Terminé !")
    print(f"   -> {OUTPUT_STAGE1} ({len(data_s1)} exemples)")
    print(f"   -> {OUTPUT_STAGE2} ({len(data_s2)} exemples)")

if __name__ == "__main__":
    convert_dataset()