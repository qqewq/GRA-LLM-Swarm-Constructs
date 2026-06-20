"""
Эксперимент: роевое взаимодействие между головами внимания как конструктами.
"""
import torch
import torch.nn.functional as F
from gra_llm.swarm_constructs import extract_constructs, foam_pair, nullify_pair, love_metric

# Допустим, у нас есть выход одного слоя трансформера: (batch, heads, seq, dim_per_head)
# Каждую голову рассматриваем как конструкт.
batch, heads, seq, dim = 1, 8, 10, 64
layer_output = torch.randn(batch, heads, seq, dim)

# Извлекаем конструкты: каждая голова представлена своим mean-вектором по seq
constructs = layer_output.mean(dim=2)  # (1, heads, dim)
head_names = [f"head_{i}" for i in range(heads)]

# Вычисляем попарную пену
for i in range(heads):
    for j in range(i+1, heads):
        zi = constructs[0, i]
        zj = constructs[0, j]
        f_pair = foam_pair(zi, zj)
        love = love_metric(zi, zj)
        print(f"{head_names[i]} <-> {head_names[j]}: foam={f_pair.item():.3f}, love={love.item():.3f}")

# Если пена выше порога, применяем обнуление
threshold = 0.5
new_constructs = constructs.clone()
for i in range(heads):
    for j in range(i+1, heads):
        if foam_pair(constructs[0,i], constructs[0,j]) > threshold:
            print(f"Nullifying {head_names[i]} and {head_names[j]}")
            new_zi, new_zj = nullify_pair(new_constructs[0,i], new_constructs[0,j], lr=0.2)
            new_constructs[0,i] = new_zi
            new_constructs[0,j] = new_zj

print("After nullification:")
for i in range(heads):
    for j in range(i+1, heads):
        print(f"{head_names[i]} <-> {head_names[j]}: foam={foam_pair(new_constructs[0,i], new_constructs[0,j]).item():.3f}")
