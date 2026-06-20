"""
Операторы T_l: критика → пересмотр.
Включают коррекцию внимания, планов, галлюцинаций.
Свойство: Φ(T_l(h)) ≤ Φ(h).
"""

import torch
import torch.nn.functional as F
from gra_llm.foam_metrics import attention_foam, plan_foam, hallucination_foam

def T_attention(h_layer, attention_weights, expected_mask, lr=0.1):
    """
    Коррекция внимания для уменьшения Φ_attn.
    Сдвигаем attention_weights в сторону expected_mask.
    """
    expected = expected_mask.to(attention_weights.device).unsqueeze(0).unsqueeze(0)
    corrected_weights = attention_weights + lr * (expected - attention_weights)
    # можно дополнительно применить softmax для нормировки
    return corrected_weights  # обновлённая матрица, которую надо подставить вместо старой

def T_plan(h_layer, plan_dist, plan_dist_next, A_l=None, steps=3, lr=0.05):
    """
    Итеративная коррекция планов на уровне l для уменьшения plan_foam.
    """
    for _ in range(steps):
        current_foam = plan_foam(plan_dist, plan_dist_next, A_l)
        if current_foam < 1e-3:
            break
        grad = torch.autograd.grad(current_foam, plan_dist, retain_graph=True)[0]
        plan_dist = plan_dist - lr * grad
    return plan_dist

def T_hallu(llm_logits, ref_logits, mix_ratio=None, threshold=0.1):
    """
    Подмешивание референсного распределения при высокой hallu-пене.
    Если mix_ratio не задан, вычисляем пропорционально текущему foam.
    """
    foam = hallucination_foam(llm_logits, ref_logits)
    if foam < threshold:
        return llm_logits
    if mix_ratio is None:
        mix_ratio = torch.clamp(foam, 0.0, 0.5)  # не более 50% замещения
    mixed_logits = (1 - mix_ratio) * llm_logits + mix_ratio * ref_logits
    return mixed_logits

def apply_T_layers(h_layers, foam_vals, cfg):
    """
    Применяет соответствующие T_l ко всем слоям, где foam > порога.
    h_layers: список состояний слоёв
    foam_vals: список значений Φ_l
    cfg: dict с параметрами
    """
    new_h = []
    for l, (h, phi) in enumerate(zip(h_layers, foam_vals)):
        if phi > cfg['foam_threshold']:
            # Здесь нужно иметь доступ к конкретным компонентам слоя,
            # в интеграции с HF это реализуется через forward hooks.
            # В прототипе возвращаем h с предупреждением.
            print(f"Applying T at layer {l}, foam={phi.item():.4f}")
            # Заглушка: можно применить простую нормализацию
            h = F.layer_norm(h, h.size()[1:])
        new_h.append(h)
    return new_h
