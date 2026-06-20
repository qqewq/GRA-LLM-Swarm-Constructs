import torch
from gra_llm.foam_metrics import foam_layer
from gra_llm.operators import T_l

def test_T_l_reduces_foam():
    h = torch.randn(1, 5, 8) * 10
    phi_before = foam_layer(h)
    h_after = T_l(h, phi_before)
    phi_after = foam_layer(h_after)
    assert phi_after <= phi_before + 1e-6, f"Foam increased: {phi_before:.4f} -> {phi_after:.4f}"
