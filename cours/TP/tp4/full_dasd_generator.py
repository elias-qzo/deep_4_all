import json
import numpy as np
import torch
import nltk
import time
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# --- CONFIGURATION ---
API_KEY = "Bearer nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"
STUDENT_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
TEACHER_API_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
TEACHER_MODEL_NAME = "openai/gpt-oss-120b"

# Nombre de requêtes API simultanées (Ajuste si Rate Limit)
MAX_WORKERS = 8 

class DASGenerator:
    def __init__(self):
        print(f"🔄 Chargement du modèle Étudiant (Local)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        
        # Client API pour les threads (Thread-safe généralement, mais on instanciera localement si besoin)
        self.base_url = TEACHER_API_URL
        self.api_key = API_KEY

    def call_teacher_api(self, prompt, temp, stage_name):
        """Fonction exécutée par les threads (CPU/Network bound)"""
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = [
            {"role": "system", "content": "You are a math expert. Solve the problem step by step. Structure your reasoning inside <reasoning> tags."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL_NAME, messages=messages, temperature=temp,
                max_tokens=1024, logprobs=True, top_logprobs=1
            )
            return {
                "success": True,
                "stage": stage_name,
                "prompt": prompt,
                "data": response.choices[0]
            }
        except Exception as e:
            # On retourne l'erreur sans planter le thread
            return {"success": False, "error": str(e)}

    def compute_sentence_score(self, prompt, previous_context, sentence):
        """Calcul GPU (Doit rester dans le thread principal)"""
        full_text = f"{prompt}\n{previous_context}{sentence}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        target_ids = input_ids.clone()
        
        prefix = f"{prompt}\n{previous_context}"
        prefix_len = len(self.tokenizer(prefix, add_special_tokens=False).input_ids)
        
        if prefix_len < input_ids.shape[1]:
            target_ids[:, :prefix_len] = -100
        else:
            return 0.0
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            return torch.exp(-outputs.loss).item()

    def run_das_logic(self, prompt, teacher_data):
        """Logique de filtrage DASD (Utilise le GPU)"""
        full_content = teacher_data.message.content
        sentences = nltk.sent_tokenize(full_content)
        kept_sentences = []
        previous_context = ""
        
        teacher_tokens = teacher_data.logprobs.content
        if not teacher_tokens: return full_content
        p_teacher_global = np.exp(np.mean([t.logprob for t in teacher_tokens]))
        
        for sent in sentences:
            p_student = self.compute_sentence_score(prompt, previous_context, sent)
            divergence = p_teacher_global - p_student
            
            if (p_teacher_global > 0.6) and (divergence > 0.15):
                kept_sentences.append(sent)
                previous_context += sent + " "
            elif (p_teacher_global > 0.6) and (abs(divergence) <= 0.15):
                kept_sentences.append(sent)
                previous_context += sent + " "
                
        final_res = " ".join(kept_sentences)
        return final_res if len(final_res) > len(full_content) * 0.4 else None

    def generate_parallel(self, data_stage1, data_stage2):
        results_stage1 = []
        results_stage2 = []
        
        # Préparation des tâches
        tasks = []
        
        print(f"🚀 Préparation des tâches...")
        # On crée une liste de tuples (prompt, temp, stage)
        for sample in data_stage1:
            tasks.append((sample['question'], 0.3, "STAGE 1"))
        for sample in data_stage2:
            tasks.append((sample['question'], 0.9, "STAGE 2"))
            
        print(f"🔥 Lancement de {len(tasks)} tâches avec {MAX_WORKERS} threads simultanés...")
        
        # Le ThreadPoolExecutor gère les appels API en parallèle
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # On soumet toutes les requêtes API
            future_to_req = {
                executor.submit(self.call_teacher_api, t[0], t[1], t[2]): t 
                for t in tasks
            }
            
            # Barre de progression
            pbar = tqdm(total=len(tasks), desc="Processing DASD")
            
            # Dès qu'une requête API revient, on la traite sur le GPU
            for future in as_completed(future_to_req):
                res = future.result()
                
                if res["success"]:
                    # C'est ici que le GPU travaille (Séquentiel, car le main thread récupère les résultats un par un)
                    prompt = res["prompt"]
                    teacher_data = res["data"]
                    stage = res["stage"]
                    
                    filtered_response = self.run_das_logic(prompt, teacher_data)
                    
                    if filtered_response:
                        entry = {
                            "conversations": [
                                {"from": "human", "value": prompt},
                                {"from": "gpt", "value": filtered_response}
                            ]
                        }
                        if stage == "STAGE 1":
                            results_stage1.append(entry)
                        else:
                            results_stage2.append(entry)
                else:
                    print(f"⚠️ Erreur API : {res.get('error')}")
                
                pbar.update(1)
            
            pbar.close()

        # Sauvegarde finale
        print(f"\n💾 Sauvegarde des résultats...")
        with open("stage1_low_temp.json", "w", encoding="utf-8") as f:
            json.dump(results_stage1, f, indent=2, ensure_ascii=False)
        print(f"✅ Stage 1 : {len(results_stage1)} exemples sauvegardés.")
        
        with open("stage2_high_temp.json", "w", encoding="utf-8") as f:
            json.dump(results_stage2, f, indent=2, ensure_ascii=False)
        print(f"✅ Stage 2 : {len(results_stage2)} exemples sauvegardés.")

if __name__ == "__main__":
    # Vérifs NLTK
    try: nltk.data.find('tokenizers/punkt_tab')
    except LookupError: 
        print("Téléchargement NLTK...")
        nltk.download('punkt'); nltk.download('punkt_tab')

    generator = DASGenerator()
    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    
    TOTAL_SAMPLES = 2000
    SPLIT_INDEX = 1000 
    
    # Sélection des données
    data_s1 = ds.select(range(0, SPLIT_INDEX))
    data_s2 = ds.select(range(SPLIT_INDEX, TOTAL_SAMPLES))
    
    # Lancement parallèle
    generator.generate_parallel(data_s1, data_s2)