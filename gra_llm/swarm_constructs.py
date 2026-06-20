import torch
from .foam_metrics import foam_layer

def extract_constructs(h_layers):
    """
    Извлекает "конструкты" как усреднённые по токенам векторы скрытых состояний каждого слоя.
    Возвращает список тензоров формы [dim] для каждого слоя.
    """
    constructs = []
    for h in h_layers:  # h: [batch, seq_len, dim]
        c = h.mean(dim=1).squeeze(0)  # усреднение по токенам, удаление batch
        constructs.append(c)
    return constructs

def foam_pair(cA, cB):
    """
    Пена взаимодействия двух конструктов: 1 - cosine_similarity.
    """
    cos = torch.nn.functional.cosine_similarity(cA, cB, dim=0)
    return (1 - cos).item()

def apply_nullification(cA, cB, cfg=None):
    """
    "Обнуление" пены между конструктами: сдвигаем cB в сторону cA.
    """
    if cfg is None:
        cfg = {'mix_alpha': 0.2}
    alpha = cfg.get('mix_alpha', 0.2)
    cB_new = (1 - alpha) * cB + alpha * cA
    return cA, cB_new
