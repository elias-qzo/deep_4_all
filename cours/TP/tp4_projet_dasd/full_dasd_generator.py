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
        self.client = OpenAI(api_key=API_KEY, base_url=TEACHER_API_URL)

    def get_teacher_response(self, prompt, temp):
        """Appel API avec température variable"""
        messages = [
            {"role": "system", "content": "You are a math expert. Solve the problem step by step. Structure your reasoning inside <reasoning> tags."},
            {"role": "user", "content": prompt}
        ]
        try:
            return self.client.chat.completions.create(
                model=TEACHER_MODEL_NAME, messages=messages, temperature=temp,
                max_tokens=1024, logprobs=True, top_logprobs=1
            ).choices[0]
        except Exception as e:
            print(f"❌ Erreur API: {e}")
            return None

    def compute_sentence_score(self, prompt, previous_context, sentence):
        """Calcul du score de l'étudiant (Probabilité géométrique)"""
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
        """Logique de filtrage DASD"""
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
            
            # Critère DAS : On garde si le Teacher est confiant ET divergence significative
            if (p_teacher_global > 0.6) and (divergence > 0.15):
                kept_sentences.append(sent)
                previous_context += sent + " "
            elif (p_teacher_global > 0.6) and (abs(divergence) <= 0.15): # Shared knowledge
                kept_sentences.append(sent)
                previous_context += sent + " "
                
        final_res = " ".join(kept_sentences)
        return final_res if len(final_res) > len(full_content) * 0.4 else None

    def generate_batch(self, dataset_slice, temperature, output_file, stage_name):
        """Génère un lot de données pour un stage spécifique"""
        print(f"\n🚀 Démarrage {stage_name} (Temp={temperature}) - {len(dataset_slice)} exemples")
        training_data = []
        
        for sample in tqdm(dataset_slice):
            prompt = sample['question']
            teacher_res = self.get_teacher_response(prompt, temp=temperature)
            
            if teacher_res:
                filtered = self.run_das_logic(prompt, teacher_res)
                if filtered:
                    training_data.append({
                        "conversations": [
                            {"from": "human", "value": prompt},
                            {"from": "gpt", "value": filtered}
                        ]
                    })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"✅ {stage_name} terminé : {len(training_data)} exemples sauvegardés dans {output_file}")

if __name__ == "__main__":
    # Vérifs NLTK
    try: nltk.data.find('tokenizers/punkt_tab')
    except LookupError: nltk.download('punkt'); nltk.download('punkt_tab')

    # Chargement global
    generator = DASGenerator()
    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
    
    # --- PARAMÈTRES DU VRAI TP ---
    TOTAL_SAMPLES = 2000
    SPLIT_INDEX = 1000 # 1000 pour Stage 1, 1000 pour Stage 2
    
    # 1. Sélection des données
    data_stage1 = ds.select(range(0, SPLIT_INDEX))
    data_stage2 = ds.select(range(SPLIT_INDEX, TOTAL_SAMPLES))
    
    # 2. Génération STAGE 1 (Basse Température)
    generator.generate_batch(
        dataset_slice=data_stage1,
        temperature=0.3, 
        output_file="stage1_low_temp.json",
        stage_name="STAGE 1"
    )
    
    # 3. Génération STAGE 2 (Haute Température)
    generator.generate_batch(
        dataset_slice=data_stage2,
        temperature=0.9, 
        output_file="stage2_high_temp.json",
        stage_name="STAGE 2"
    )