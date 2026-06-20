import torch

def T_l(h, Phi_l, cfg=None):
    """
    Оператор обнуления пены на уровне скрытого состояния.
    Простейшая реализация: ограничение нормы активаций (clamping).
    """
    if cfg is None:
        cfg = {'clamp_value': 5.0}
    clamp_val = cfg.get('clamp_value', 5.0)
    h_clamped = torch.clamp(h, -clamp_val, clamp_val)
    return h_clamped
