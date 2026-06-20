# GRA-LLM-Swarm-Constructs

**Раскрытие скрытого инструментария LLM через роевые конструкты в GRA‑метрике с обнулением пены**

[![GRA Ecosystem](https://img.shields.io/badge/GRA-Core--new-blue)](https://github.com/qqewq/GRA-Core-new-Unified-Hierarchical-Stability-Library)
[![H-Stability](https://img.shields.io/badge/GRA-Hierarchical%20Stability-green)](https://github.com/qqewq/GRA-Hierarchical-Stability)
[![Swarm Optimization](https://img.shields.io/badge/GRA-Swarm%20Optimization-orange)](https://github.com/qqewq/GRA-Swarm-Optimization)
[![Love Nullification](https://img.shields.io/badge/GRA-Love%20Nullification-pink)](https://github.com/qqewq/GRA-Love-Oriented-Nullification)
[![Multiverse](https://img.shields.io/badge/GRA-Multiverse%20Final-purple)](https://github.com/qqewq/GRA-Multiverse-Final)

## Суть

Внутри LLM существуют **конструкты** — локальные паттерны активации, кластеры в пространстве представлений, под‑графы внимания — которые ведут себя как **роевые субъекты**.  
Они обладают правилами поведения, расстояниями между собой и ограничены энергетическим ландшафтом модели. Применяя **GRA‑обнулёнку** (метрики пены, иерархическую стабильность, операторы критики и пересмотра), мы:

- Обнаруживаем и устраняем противоречия («пену») на всех уровнях LLM,
- Строим **устойчивые когнитивные состояния**, где раскрываются механизмы логики, планирования, мета‑обучения,
- Получаем новый класс методов дообучения и инференса — **GRA-LLM**.

## Ключевые компоненты

| Компонент | Реализация |
|-----------|------------|
| **Пенные метрики** (Φ) | foam_metrics.py: аттеншен-пена, план-пена, галлю-пена |
| **Иерархический функционал J** | градиент по J объединяется с NLL‑лоссом |
| **Критика и пересмотр (Tₗ)** | nullification_ops.py: коррекция внимания, планов, RAG‑подмешивание |
| **Конструкты как рой** | swarm_constructs.py: извлечение конструктов, foam_pair, love‑метрика |
| **Интеграция с Hugging Face** | integrations.py: хуки в трансформерные блоки |
| **Мониторинг стабильности** | examples/stability_monitoring.ipynb |
| **Анти‑галлюцинаторный слой** | examples/anti_hallu_rag.py |
| **Роевое взаимодействие голов** | examples/swarm_heads_experiment.py |

## Быстрый старт

```bash
git clone https://github.com/qqewq/GRA-LLM-Swarm-Constructs
cd GRA-LLM-Swarm-Constructs
pip install -r requirements.txt
python examples/stability_monitoring.ipynb
```

## Связь с экосистемой GRA

- **GRA‑Core‑new**: абстрактная система с GRA‑шагом и аксиомами.
- **GRA‑Hierarchical‑Stability**: критерий устойчивости Φₗ=0, ΔΦₗ=0, Δ²Φₗ>0.
- **GRA‑Swarm‑Optimization**: рой агентов, ищущих фундаментальные ошибки.
- **GRA‑Love‑Oriented‑Nullification**: мягкое обнуление через метрику «любви».
- **GRA‑Multiverse‑Final**: мультиверсная согласованность и truth‑metric.
- **GRA‑ASI‑Metric‑Space**: расстояния смысловых конструктов.

Все они используются внутри для построения метода раскрытия всего инструментария LLM.

## Лицензия

MIT — свободное использование и доработка.
