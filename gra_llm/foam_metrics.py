import torch
import torch.nn.functional as F

def foam_layer(h):
    """
    Слойная пена: энтропия распределения активаций скрытого состояния.
    h: [batch, seq_len, dim]
    Возвращает скаляр Phi_l.
    """
    probs = F.softmax(h, dim=-1)
    log_probs = F.log_softmax(h, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch, seq_len]
    return entropy.mean().item()

def foam_attn(attn, attn_ref=None):
    """
    Пена внимания: KL-дивергенция между распределением внимания и равномерным (или эталонным).
    attn: [batch, heads, seq_q, seq_k] --- уже после softmax.
    """
    if attn_ref is None:
        attn_ref = torch.ones_like(attn) / attn.size(-1)
    kl = torch.sum(attn * (torch.log(attn + 1e-9) - torch.log(attn_ref + 1e-9)), dim=-1)
    return kl.mean().item()

def foam_hallu(p_model, p_ref):
    """
    Пена галлюцинаций: KL-дивергенция между распределением токенов модели и эталонным.
    p_model, p_ref: [batch, vocab] вероятности.
    """
    kl = torch.sum(p_ref * (torch.log(p_ref + 1e-9) - torch.log(p_model + 1e-9)), dim=-1)
    return kl.mean().item()
