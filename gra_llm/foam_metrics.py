"""
Пенные метрики Φ для уровней LLM.
Основаны на GRA-Hierarchical-Stability и GRA-Core-new.
"""

import torch
import torch.nn.functional as F

def attention_foam(attention_weights, expected_mask=None):
    """
    Φ_attn: среднеквадратичное отклонение от ожидаемой структуры внимания.
    attention_weights: (batch, heads, seq, seq)
    expected_mask: (seq, seq) или (1,1,seq,seq) — желаемая маска (0/1 или вероятности)
    """
    if expected_mask is None:
        # По умолчанию: каузальность + локальный контекст (пример)
        seq_len = attention_weights.shape[-1]
        expected_mask = torch.tril(torch.ones(seq_len, seq_len))  # causal
        # Можно добавить затухание по расстоянию
    # Приводим expected_mask к форме attention_weights
    expected = expected_mask.to(attention_weights.device).unsqueeze(0).unsqueeze(0)
    diff = attention_weights - expected
    return torch.mean(diff ** 2)

def plan_foam(plan_dist_l, plan_dist_l_plus_1, A_l=None):
    """
    Φ_plan: KL-дивергенция между планами соседних уровней после подъёма.
    plan_dist_l: (batch, seq, dim) — логиты/вероятности планов на уровне l
    plan_dist_l_plus_1: (batch, seq, dim)
    A_l: оператор подъёма (линейный слой или identity)
    """
    if A_l is not None:
        plan_dist_l_plus_1 = A_l(plan_dist_l_plus_1)
    # Превращаем в вероятности
    p_l = F.softmax(plan_dist_l, dim=-1)
    p_lp1 = F.softmax(plan_dist_l_plus_1, dim=-1)
    # KL(p_l || p_lp1) в среднем по batch и позициям
    kl = F.kl_div(p_l.log(), p_lp1, reduction='batchmean', log_target=False)
    return kl

def hallucination_foam(llm_logits, ref_logits, d='js'):
    """
    Φ_hallu: расхождение между выходом LLM и референсным распределением (RAG/верификатор).
    llm_logits: (batch, seq, vocab)
    ref_logits: (batch, seq, vocab)
    d: 'kl' или 'js'
    """
    p_llm = F.softmax(llm_logits, dim=-1)
    p_ref = F.softmax(ref_logits, dim=-1)
    if d == 'js':
        m = 0.5 * (p_llm + p_ref)
        js = 0.5 * (F.kl_div(p_llm.log(), m, reduction='batchmean', log_target=False) +
                    F.kl_div(p_ref.log(), m, reduction='batchmean', log_target=False))
        return js
    else:
        return F.kl_div(p_llm.log(), p_ref, reduction='batchmean', log_target=False)

def composite_foam(h_layer, attention_weights, plan_dist=None, plan_next=None,
                   llm_logits=None, ref_logits=None, weights=(1.0,1.0,1.0)):
    """
    Собирает общую пену слоя.
    weights: (w_attn, w_plan, w_hallu)
    """
    foam = 0.0
    if weights[0] > 0 and attention_weights is not None:
        foam += weights[0] * attention_foam(attention_weights)
    if weights[1] > 0 and plan_dist is not None and plan_next is not None:
        foam += weights[1] * plan_foam(plan_dist, plan_next)
    if weights[2] > 0 and llm_logits is not None and ref_logits is not None:
        foam += weights[2] * hallucination_foam(llm_logits, ref_logits)
    return foam
