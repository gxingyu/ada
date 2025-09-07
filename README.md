# Adaptive Distribution Alignment via Characteristic Function for Graph Domain  Adaptation

This is the source code of "Adaptive Distribution Alignment via Characteristic Function for Graph Domain  Adaptation"

# Abstract

Graph Domain Adaptation (GDA) aims to transfer knowledge from labeled source graphs to unlabeled target graphs, where the core challenge arises from cross-domain graph distribution shifts. While existing methods typically focus on isolated aspects of distribution misalignment, such as node attributes, degree distributions, or graph heterophily, they often overlook the interdependence of these multi-faceted shifts, leading to suboptimal adaptation. To bridge this gap, we propose a unified distribution matching framework that holistically quantifies and aligns composite graph shifts. Our key insight is that graph distributions can be uniquely characterized via neural characteristic functions, which inherently encapsulate joint feature-structure dependencies through Fourier-domain representations. By reformulating GDA as a minimax optimization problem, where a learnable discrepancy metric dynamically prioritizes dominant shifts, our framework achieves adaptive alignment without manual reweighting. Theoretical analysis and empirical evaluations demonstrate that our approach outperforms state-of-the-art methods in handling complex, coupled distribution shifts.

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
