"""
Мониторинг пены Phi_l, dPhi_l, d2Phi_l на слоях LLM (TinyLlama).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gra_llm.foam_metrics import foam_layer

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cpu"
    model.to(device)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hidden_states = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())

    layers = model.model.layers
    handles = [layer.register_forward_hook(hook) for layer in layers]

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    phis = [foam_layer(h) for h in hidden_states]
    dphis = [None] + [phis[i] - phis[i-1] for i in range(1, len(phis))]
    d2phis = [None, None] + [dphis[i] - dphis[i-1] for i in range(2, len(dphis))]

    print(f"\n{'Layer':<6} {'Phi':>10} {'dPhi':>10} {'d2Phi':>10}")
    print("-" * 40)
    for i in range(len(phis)):
        dphi_str = f"{dphis[i]:10.4f}" if dphis[i] is not None else "       -"
        d2phi_str = f"{d2phis[i]:10.4f}" if d2phis[i] is not None else "       -"
        print(f"{i:<6} {phis[i]:10.4f} {dphi_str} {d2phi_str}")

if __name__ == "__main__":
    main()
