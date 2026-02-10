import json
import numpy as np
import torch
import nltk
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- CONFIGURATION ---
API_KEY = "Bearer nKuJabWS1epvq3x-m8by6NOU4xP4_znNL9OhmgXBPz9OeWOHlyGJIENnG8oXLT-4oOXNmESqExEMZv6o"  
STUDENT_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
TEACHER_API_URL = "https://api.infomaniak.com/2/ai/48/openai/v1"
TEACHER_MODEL_NAME = "openai/gpt-oss-120b" 

# Fichier de sortie
OUTPUT_FILE = "train_dasd_gsm8k.json"

class DASGenerator:
    def __init__(self):
        print(f"🔄 Chargement du modèle Étudiant (Local) : {STUDENT_MODEL_ID}...")
        # Configuration 4-bit pour que ça tienne sur ton GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        print("✅ Modèle étudiant chargé.")
        self.client = OpenAI(api_key=API_KEY, base_url=TEACHER_API_URL)

    def get_teacher_response(self, prompt, temp=0.7):
        """Récupère la réponse et les logprobs du Teacher"""
        messages = [
            {
                "role": "system", 
                # Prompt spécifique pour GSM8K (Maths)
                "content": "You are a math expert. Solve the problem step by step. Structure your reasoning inside <reasoning> tags before giving the final answer."
            },
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=TEACHER_MODEL_NAME,
                messages=messages,
                temperature=temp,
                max_tokens=1024, # Suffisant pour des maths
                logprobs=True,
                top_logprobs=1
            )
            return response.choices[0]
        except Exception as e:
            print(f"❌ Erreur API: {e}")
            return None

    def compute_sentence_score(self, prompt, previous_context, sentence):
        """
        Calcule la probabilité (P_s) que l'étudiant génère cette phrase.
        """
        full_text = f"{prompt}\n{previous_context}{sentence}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        input_ids = inputs.input_ids
        target_ids = input_ids.clone()
        
        # On masque le prompt et le contexte précédent pour ne calculer la loss QUE sur la phrase actuelle
        prefix = f"{prompt}\n{previous_context}"
        prefix_tokens = self.tokenizer(prefix, add_special_tokens=False).input_ids
        prefix_len = len(prefix_tokens)
        
        # Si le préfixe est plus long que l'input (cas rare d'erreur de tokenisation), on sécurise
        if prefix_len < input_ids.shape[1]:
            target_ids[:, :prefix_len] = -100
        else:
            return 0.0 # Fallback
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            loss = outputs.loss
            
        # P_student = exp(-loss)
        return torch.exp(-loss).item()

    def run_das_logic(self, prompt, teacher_data):
        """Applique l'algorithme DAS phrase par phrase"""
        full_content = teacher_data.message.content
        
        # Découpage en phrases (mieux pour le raisonnement mathématique)
        sentences = nltk.sent_tokenize(full_content)
        
        kept_sentences = []
        previous_context = ""
        
        # Récupération globale de la confiance du Teacher (approximation via API)
        teacher_tokens = teacher_data.logprobs.content
        if not teacher_tokens:
            return full_content # Si pas de logprobs, on garde tout par sécurité
            
        teacher_logprobs = [t.logprob for t in teacher_tokens]
        p_teacher_global = np.exp(np.mean(teacher_logprobs))
        
        for sent in sentences:
            # 1. Score Etudiant (Est-ce qu'il aurait pu dire ça ?)
            p_student = self.compute_sentence_score(prompt, previous_context, sent)
            
            # 2. Score Teacher (On utilise la moyenne globale ici pour simplifier l'alignement token/phrase)
            p_teacher = p_teacher_global 
            
            # --- LOGIQUE DE FILTRAGE DAS ---
            # On cherche les phrases où le Prof est SÛR mais l'élève NE SAIT PAS.
            divergence = p_teacher - p_student
            
            # Critères (ajustables selon tes résultats)
            is_teacher_sentence = (p_teacher > 0.6) and (divergence > 0.15)
            is_shared_sentence = (p_teacher > 0.6) and (abs(divergence) <= 0.15)
            
            if is_teacher_sentence or is_shared_sentence:
                kept_sentences.append(sent)
                previous_context += sent + " "
            else:
                # Si c'est une "Student Sentence" (hallucination probable de l'élève si on le laissait faire)
                # On ne l'ajoute pas au dataset d'entrainement.
                pass

        # Reconstruct filtered response
        final_response = " ".join(kept_sentences)
        
        # Si on a trop filtré (réponse vide ou incohérente), on ignore cet exemple
        if len(final_response) < 10: 
            return None
            
        return final_response

    def generate_dataset(self, num_samples=10):
        print(f"📥 Téléchargement du dataset GSM8K...")
        ds = load_dataset("gsm8k", "main", split="train")
        
        training_data = []  # Pour Llama-Factory (Format ShareGPT)
        analysis_data = []  # Pour ton rapport (Avec les stats)
        
        print(f"🚀 Démarrage de la génération pour {num_samples} exemples...")
        
        shuffled_ds = ds.shuffle(seed=42).select(range(num_samples * 2))
        
        valid_count = 0
        pbar = tqdm(total=num_samples)
        
        for sample in shuffled_ds:
            if valid_count >= num_samples:
                break
                
            prompt = sample['question']
            
            # 1. Appel Teacher
            teacher_res = self.get_teacher_response(prompt)
            if not teacher_res: continue
            
            # 2. Application DAS (On récupère maintenant le texte ET les stats)
            # Note: J'ai adapté run_das_logic pour qu'il retourne aussi les scores si possible
            # Pour faire simple ici, on va stocker les stats globales du teacher
            
            filtered_response = self.run_das_logic(prompt, teacher_res)
            
            if filtered_response:
                # --- Fichier pour l'entrainement (PROPRE) ---
                train_entry = {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": filtered_response}
                    ]
                }
                training_data.append(train_entry)
                
                # --- Fichier pour l'analyse (RICHE) ---
                # On récupère les logprobs bruts du teacher pour l'analyse
                teacher_logprobs = [t.logprob for t in teacher_res.logprobs.content] if teacher_res.logprobs else []
                mean_logprob = np.mean(teacher_logprobs) if teacher_logprobs else 0.0
                
                analysis_entry = {
                    "prompt": prompt,
                    "full_teacher_response": teacher_res.message.content,
                    "filtered_response": filtered_response,
                    "stats": {
                        "teacher_mean_logprob": float(mean_logprob),
                        "teacher_total_logprob": float(np.sum(teacher_logprobs)),
                        "length_original": len(teacher_res.message.content),
                        "length_filtered": len(filtered_response)
                    }
                }
                analysis_data.append(analysis_entry)

                valid_count += 1
                pbar.update(1)
        
        pbar.close()
        
        # Sauvegarde 1 : Pour Llama-Factory
        with open("train_dasd_gsm8k.json", "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
            
        # Sauvegarde 2 : Pour ton rapport (Python/Pandas)
        with open("analysis_dasd.json", "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Terminé !")
        print(f"   -> Dataset d'entraînement : train_dasd_gsm8k.json (Utiliser pour Llama-Factory)")
        print(f"   -> Données d'analyse : analysis_dasd.json (Utiliser pour les graphiques du rapport)")

if __name__ == "__main__":
    # Vérification NLTK
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("⬇️ Téléchargement des ressources NLTK manquantes...")
        nltk.download('punkt')
        nltk.download('punkt_tab')

    generator = DASGenerator()
    generator.generate_dataset(num_samples=2000)