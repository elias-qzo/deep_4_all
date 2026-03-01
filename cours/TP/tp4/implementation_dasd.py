import os
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class DASPipelineQwen:
    def __init__(self, student_model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.student_model_id = student_model_id
        print(f"Chargement du modèle étudiant : {self.student_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        api_key = os.environ.get("INFOMANIAK_API_KEY")
        if not api_key:
            raise ValueError("La variable d'environnement INFOMANIAK_API_KEY n'est pas définie.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.infomaniak.com/2/ai/48/openai/v1")
        self.teacher_model_name = "openai/gpt-oss-120b"

    def get_teacher_data(self, user_prompt, temperature=0.7):
        messages = [{"role": "user", "content": user_prompt}]
        response = self.client.chat.completions.create(
            model=self.teacher_model_name,
            messages=messages,
            temperature=temperature,
            logprobs=True,
            top_logprobs=1,
        )

        content = response.choices[0].message.content
        logprobs_data = response.choices[0].logprobs

        if not logprobs_data:
            raise ValueError("L'API Teacher n'a pas renvoyé de logprobs.")

        tokens = [t.token for t in logprobs_data.content]
        logprobs = [t.logprob for t in logprobs_data.content]

        mean_logprob = np.exp(np.mean(logprobs)) if logprobs else 0.0

        return {
            "content": content,
            "tokens": tokens,
            "logprobs": logprobs,
            "total_logprob": sum(logprobs),
            "mean_logprob": mean_logprob,
            "num_tokens": len(tokens),
        }

    def get_student_logprobs(self, prompt: str, response: str) -> dict:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        response_start_idx = prompt_tokens.shape[1]

        labels = input_ids.clone()
        labels[:, :response_start_idx] = -100

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            token_logprobs = -token_losses

            valid_mask = shift_labels != -100
            valid_logprobs = token_logprobs[valid_mask].cpu().numpy()

        mean_logprob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0

        return {
            "total_logprob": float(np.sum(valid_logprobs)),
            "mean_logprob": mean_logprob,
            "num_tokens": len(valid_logprobs),
            "logprobs": valid_logprobs.tolist(),
        }

    def decide_keep_prompt(self, teacher_answer, student_answer):
        p_teacher = teacher_answer.get("mean_logprob", 0.0)
        p_student = student_answer.get("mean_logprob", 0.0)
        divergence = p_teacher - p_student
        return (p_teacher > 0.6) and (divergence > 0.15)

    def run(self, prompt):
        print(f"Traitement : '{prompt[:80]}...'")

        teacher_answer = self.get_teacher_data(prompt)
        if not teacher_answer:
            return None

        print(f"Réponse Teacher reçue ({len(teacher_answer['content'])} chars).")

        try:
            student_answer = self.get_student_logprobs(prompt, teacher_answer["content"])
            keep = self.decide_keep_prompt(teacher_answer, student_answer)
            print(f"DAS : P_teacher={teacher_answer['mean_logprob']:.3f}, P_student={student_answer['mean_logprob']:.3f} → {'GARDER' if keep else 'REJETER'}")
            return keep
        except Exception as e:
            print(f"Erreur DAS : {e}")
            return None


if __name__ == "__main__":
    STUDENT_ID = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

    pipeline = DASPipelineQwen(student_model_id=STUDENT_ID)

    test_prompt = "Explique le principe de la supraconductivité de manière simple."
    result = pipeline.run(test_prompt)
    print(f"Résultat final : {'Exemple conservé' if result else 'Exemple rejeté'}")
