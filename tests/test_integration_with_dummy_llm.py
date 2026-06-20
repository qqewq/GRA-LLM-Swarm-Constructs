import torch
import torch.nn as nn
from gra_llm.integration import gra_llm_step

class DummyLLM(nn.Module):
    def __init__(self, vocab_size=10, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layer = nn.Linear(dim, dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        # Имитация структуры transformers
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self.layer])

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.layer(x)
        logits = self.lm_head(x)
        return type('output', (), {'logits': logits})()

def test_gra_step_returns_expected_keys():
    model = DummyLLM()
    input_ids = torch.randint(0, 10, (1, 4))
    result = gra_llm_step(model, input_ids)
    assert 'logits' in result
    assert 'phi_values' in result
    assert 'max_phi_layer' in result
    assert result['logits'].shape == (1, 4, 10)

def test_max_phi_not_exploding():
    model = DummyLLM()
    input_ids = torch.randint(0, 10, (1, 4))
    res = gra_llm_step(model, input_ids)
    assert max(res['phi_values']) < 100.0
