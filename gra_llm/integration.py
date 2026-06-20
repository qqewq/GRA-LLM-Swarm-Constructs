import torch
from .foam_metrics import foam_layer
from .operators import T_l

def gra_llm_step(model, input_ids, gra_cfg=None):
    """
    Выполняет один GRA-шаг для языковой модели:
    - перехватывает скрытые состояния всех слоёв,
    - вычисляет Phi_l,
    - находит слой с максимальной пеной,
    - применяет T_l к этому состоянию (демонстрационно).
    Возвращает словарь с результатами анализа.
    """
    if gra_cfg is None:
        gra_cfg = {}

    hidden_states = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states.append(output[0].detach())
        else:
            hidden_states.append(output.detach())

    # Определяем слои модели (поддержка LLaMA и GPT-2 подобных)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Неизвестная архитектура модели. Добавьте свой способ получения слоёв.")

    handles = [layer.register_forward_hook(hook) for layer in layers]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits = outputs.logits

    for h in handles:
        h.remove()

    phi_values = [foam_layer(hs) for hs in hidden_states]
    max_phi_layer = max(range(len(phi_values)), key=lambda i: phi_values[i])
    max_phi = phi_values[max_phi_layer]

    hs_max = hidden_states[max_phi_layer]
    hs_corrected = T_l(hs_max, max_phi, gra_cfg.get('T_cfg', {}))
    phi_after = foam_layer(hs_corrected)

    return {
        'logits': logits,
        'phi_values': phi_values,
        'max_phi_layer': max_phi_layer,
        'phi_before': max_phi,
        'phi_after': phi_after,
        'hidden_states': hidden_states
    }
