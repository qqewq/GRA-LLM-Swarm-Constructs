"""
Хуки для интеграции GRA-LLM в Hugging Face трансформеры.
Позволяет перехватывать скрытые состояния и внимание на каждом слое.
"""

import torch
from transformers import PreTrainedModel
from gra_llm.foam_metrics import composite_foam
from gra_llm.nullification_ops import apply_T_layers

class GRAWrapper:
    def __init__(self, model: PreTrainedModel, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.layer_outputs = []   # будет хранить (hidden_states, attention_weights)
        self._register_hooks()

    def _register_hooks(self):
        # Регистрируем forward hooks на все encoder/decoder слои
        for i, layer in enumerate(self.model.base_model.encoder.layer):
            layer.register_forward_hook(self._hook(i), with_kwargs=True)

    def _hook(self, layer_idx):
        def hook(module, input, output):
            # output обычно tuple: (hidden_states, ...) или (hidden_states, attention, ...)
            # для простоты предполагаем output[0] - hidden, output[1] - attention
            if isinstance(output, tuple):
                hidden = output[0]
                attn = output[1] if len(output) > 1 else None
            else:
                hidden = output
                attn = None
            self.layer_outputs.append((layer_idx, hidden, attn))
        return hook

    def forward_with_gra(self, input_ids, attention_mask=None, **kwargs):
        self.layer_outputs = []
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, output_attentions=True, **kwargs)
        # Собираем пену
        hidden_states = outputs.hidden_states  # tuple для всех слоёв
        attentions = outputs.attentions        # tuple
        foam_vals = []
        for l in range(len(hidden_states)):
            h = hidden_states[l]
            attn = attentions[l] if attentions is not None else None
            # Вычисляем пену; ref_logits=None пока нет RAG
            phi = composite_foam(h, attn, weights=(1.0,0,0))
            foam_vals.append(phi)
        # Проверяем устойчивость и применяем T_l
        # (упрощённо: пороговая коррекция)
        if self.cfg.get('apply_nullification', True):
            hidden_states = apply_T_layers(hidden_states, foam_vals, self.cfg)
            # Заново прогоняем head? В реальности нужно обновить hidden_states в модели.
            # Здесь для примера возвращаем logits без изменений.
        return outputs, foam_vals
