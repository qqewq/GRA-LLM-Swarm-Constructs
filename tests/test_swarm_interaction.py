import torch
from gra_llm.swarm_constructs import foam_pair, apply_nullification

def test_nullification_reduces_pair_foam():
    cA = torch.tensor([1.0, 0.0, 0.0])
    cB = torch.tensor([0.0, 1.0, 0.0])
    phi_before = foam_pair(cA, cB)
    _, cB_new = apply_nullification(cA, cB, {'mix_alpha': 0.5})
    phi_after = foam_pair(cA, cB_new)
    assert phi_after < phi_before, f"Foam did not decrease: {phi_before:.4f} -> {phi_after:.4f}"
