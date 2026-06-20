"""
Конструкты как роевые субъекты. Извлечение, метрики расстояния, foam_pair, Love-метрика.
Интеграция с GRA-Swarm-Optimization, GRA-Love-Oriented-Nullification,
GRA-ASI-Metric-Space.
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans  # для прототипа

def extract_constructs(layer_hidden, num_constructs=4):
    """
    Простейшее извлечение конструктов: кластеризация векторов активаций.
    layer_hidden: (batch, seq, hidden_dim) -> (batch*seq, hidden_dim)
    Возвращает список состояний конструктов z_i (центроидов) и индексы.
    В реальности можно обучать через EM или attention.
    """
    flat = layer_hidden.view(-1, layer_hidden.size(-1)).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_constructs, random_state=0).fit(flat)
    centers = torch.tensor(kmeans.cluster_centers_, device=layer_hidden.device)
    return centers  # (num_constructs, hidden_dim)

def gra_distance(z_i, z_j, metric='cosine'):
    """
    GRA-метрика расстояния между конструктами.
    Из GRA-ASI-Metric-Space.
    """
    if metric == 'cosine':
        return 1.0 - F.cosine_similarity(z_i, z_j, dim=-1)
    else:
        return torch.norm(z_i - z_j, p=2, dim=-1)

def foam_pair(z_i, z_j, attention_pair=None, threshold=0.5):
    """
    Пена пары конструктов. Может учитывать:
    - расхождение состояний,
    - противоречия в их внимании,
    - взаимную информацию.
    """
    # Базово: расстояние + добавочный штраф если есть attention_pair
    dist = gra_distance(z_i, z_j)
    foam = dist
    if attention_pair is not None:
        # attention_pair: (seq, seq) или скаляр агрегации
        foam += 0.1 * attention_pair.mean()  # пример
    return foam

def love_metric(z_i, z_j, history_alignment=None):
    """
    Love-метрика L_ij из GRA-Love-Oriented-Nullification.
    0 — полный конфликт, 1 — абсолютная гармония.
    Можно вычислять как косинусное сходство плюс исторический контекст.
    """
    cos_sim = F.cosine_similarity(z_i, z_j, dim=-1)
    L = (cos_sim + 1.0) / 2.0  # в [0,1]
    if history_alignment is not None:
        L = 0.7 * L + 0.3 * history_alignment  # сглаживание
    return L

def nullify_pair(z_i, z_j, lr=0.1, love=None):
    """
    Мягкое обнуление конфликта пары.
    При высокой любви — слабая коррекция, при низкой — сильная.
    Возвращает новые z_i, z_j.
    """
    if love is None:
        love = love_metric(z_i, z_j)
    # Сила коррекции обратно пропорциональна любви
    strength = lr * (1.0 - love)
    direction = (z_j - z_i)  # условно: двигаем к гармонии
    z_i_new = z_i + strength * direction
    z_j_new = z_j - strength * direction
    return z_i_new, z_j_new
