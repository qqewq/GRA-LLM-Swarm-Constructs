"""
Один GRA-шаг: анализ до и после применения T_l к слою с макс. пеной.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gra_llm.integration import gra_llm_step

def generate_text(model, tokenizer, input_ids, max_new_tokens=20):
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cpu"
    model.to(device)

    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    original_text = generate_text(model, tokenizer, inputs.input_ids)
    print("Original answer:", original_text)

    result = gra_llm_step(model, inputs.input_ids)
    print(f"Max foam layer: {result['max_phi_layer']}, Phi before: {result['phi_before']:.4f}, after T_l: {result['phi_after']:.4f}")
    print("(При реальной коррекции скрытого состояния ответ мог бы измениться.)")

if __name__ == "__main__":
    main()
