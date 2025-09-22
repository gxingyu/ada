# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

This is the source code of "Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation"

# Abstract

Graph Domain Adaptation (GDA) transfers knowledge from labeled source graphs to unlabeled target graphs but is challenged by complex, multi-faceted distributional shifts. Existing methods attempt to reduce distributional shifts through heuristic alignment strategies tailored to specific graph elements (e.g., node attributes or structural statistics), which typically require manually designed graph filters to extract relevant features before alignment. However, such handcrafted approaches are inherently rigid: they rely on task-specific heuristics, and struggle when dominant discrepancies vary across transfers.
To address these limitations,
we propose ADAlign, an Adaptive Distribution Alignment framework for GDA. Unlike heuristic methods, ADAlign requires no manual specification of alignment criteria. It automatically identifies the most relevant discrepancies in each transfer and aligns them jointly, capturing the interplay between attributes, structures, and their dependencies. This makes ADAlign flexible, task-aware, and robust to diverse and dynamically evolving shifts.
To enable this adaptivity, we introduce the Neural Spectral Discrepancy (NSD), a theoretically principled parametric distance that provides a unified view of cross-graph shifts. NSD leverages neural characteristic functions in the spectral domain to encode feature-structure dependencies of all orders, while a learnable frequency sampler adaptively emphasizes the most informative spectral components for each task within minmax paradigm.
Extensive experiments on 10 datasets and 16 transfer tasks show that ADAlign not only outperforms state-of-the-art baselines but also achieves efficiency gains with lower memory usage and faster training.

# Requirements

This code requires the following:

* torch==2.4.1
* torch-scatter==2.1.2
* torch-sparse==0.6.18
* torch-cluster==1.6.3
* torch-geometric==2.6.1
* numpy==1.26.4
* pygda==1.2.0
* scikit-learn==1.6.1

# Dataset

* **Airport**: It has 3 different domains, i.e., Brazil, Euroup and USA. They can be
  downloaded [here](https://drive.google.com/drive/folders/1zlluWoeukD33ZxwaTRQi3jCdD0qC-I2j?usp=share_link). The graph
  processing can be found at ``AirportDataset``. We utilize ``OneHotDegree`` to construct node features for each node.
* **Blog**: It has 2 different domains, i.e., Blog1 and Blog2. They can be
  downloaded [here](https://drive.google.com/drive/folders/1jKKG0o7rEY-BaVEjBhuGijzwwhU0M-pQ?usp=share_link). The graph
  processing can be found at ``BlogDataset``.
* **Citation**: It has 3 different domains, i.e., ACMv9 , Citationv1 and DBLPv7. They can be
  downloaded [here](https://drive.google.com/drive/folders/1ntNt3qHE4p9Us8Re9tZDaB-tdtqwV8AX?usp=share_link). The graph
  processing can be found at ``CitationDataset``.
* **Twitch**: It has 6 different domains, i.e., DE, EN, ES, FR,PT and RU. They are adopted from the [Twitch Social Networks](https://github.com/benedekrozemberczki/datasets?tab=readme-ov-file#twitch-social-networks) and can be downloaded [here](https://drive.google.com/drive/folders/1GWMyyJOZ4CeeqP_H5dCA5voSQHT0WlXG?usp=share_link). The graph processing can be found at `TwitchDataset`.

# Training: How to run the code

```python
python main.py --source <source dataset> --target <target dataset>
```

# Evaluate

The `eval_micro_f1` method from the `pygda` library can easily be used to evaluate the performance of the model.

```python
logits, labels = model.predict(target_data)
preds = logits.argmax(dim=1)
mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)
print('micro-f1: ' + str(mi_f1))
print('macro-f1: ' + str(ma_f1))
```
