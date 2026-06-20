"""
Обнаружение галлюцинаций через пену и принудительная коррекция с помощью фактов.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gra_llm.foam_metrics import foam_hallu

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cpu"
    model.to(device)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    # Эталонное распределение: токен "Paris" имеет вероятность 1
    ref_probs = torch.zeros_like(probs)
    paris_id = tokenizer.encode("Paris", add_special_tokens=False)[0]
    ref_probs[paris_id] = 1.0

    hallu_foam = foam_hallu(probs.unsqueeze(0), ref_probs.unsqueeze(0))
    print(f"Hallucination foam: {hallu_foam:.4f}")

    if hallu_foam > 1.0:
        print("Высокая пена галлюцинаций! Используем факт из базы знаний.")
        print("Исправленный ответ:", prompt + " Paris")
    else:
        predicted_id = torch.argmax(probs).item()
        predicted_token = tokenizer.decode([predicted_id])
        print("Ответ модели:", prompt + " " + predicted_token)

if __name__ == "__main__":
    main()
