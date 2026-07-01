# Santander Customer Transaction Prediction

Binary classification of anonymized Santander transaction features with severe class imbalance (~10:1 ratio).

## Results

| Model | OOF AUROC |
|---|---|
| LightGBM Baseline (10-fold CV) | 0.8947 |
| Neural Network (10-fold CV) | 0.8410 |
| LightGBM Master (pseudo-labeling only) | 0.8990 |
| **LGBM Student-Master Pipeline** | **0.9520** |
| LightGBM + NN Blend (50/50) | 0.8929 |


## Technical Highlights

- **Unseen-count feature engineering**: exploits the fact that Kaggle's test set contains synthetic rows — values that never appear in training. Counting these per row creates a powerful meta-feature.
- **Shuffle augmentation**: 16× positive, 4× negative duplication with shuffled feature values produces ~1M synthesized rows, dramatically expanding minority-class representation.
- **Student → Pseudo-Label → Master pipeline**: Student model trained on augmented data generates confident test predictions. Top-2700 positives and top-2000 negatives added as pseudo-labels to the master training set. OOF AUROC: 0.8947 → 0.9520.
- **RankGauss normalization**: QuantileTransformer (output_distribution='normal') + StandardScaler applied to neural network features. Stabilises training and improves convergence.


## Repository Structure

```
santander-transaction-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md              ← Kaggle download instructions (no CSVs committed)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── lightgbm_baseline/
│   │   ├── config.py
│   │   ├── preprocess.py      ← unseen-count, freq encoding, rank transform
│   │   └── train.py           ← 10-fold CV, OOF AUROC 0.8947
│   ├── neural_network/
│   │   ├── config.py
│   │   ├── preprocess.py      ← same features + StandardScaler
│   │   ├── model.py           ← 3-layer MLP with BatchNorm + Dropout
│   │   └── train.py           ← 10-fold CV, OOF AUROC 0.8410
│   ├── pseudo_labeling_pipeline/
│   │   ├── config.py
│   │   ├── preprocess.py
│   │   └── train.py           ← Student→Master, OOF AUROC 0.9520
│   └── ensemble/
│       ├── lgbm_catboost/
│       │   ├── config.py
│       │   ├── preprocess.py
│       │   └── train.py       ← LGBM + CatBoost 50/50 blend
│       └── lgbm_nn/
│           ├── config.py
│           ├── preprocess.py
│           ├── model.py       ← per-model data prep + NN architecture
│           └── train.py       ← LGBM + NN blend, OOF AUROC 0.8929
└── results/
    ├── metrics.md
    └── model_comparison.png   ← add after running experiments
```

## How to Run

### 1. Set up the environment

```bash
pip install -r requirements.txt
```

### 2. Download the data

See [data/README.md](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data) for Kaggle CLI instructions.

### 3. Run a model

Each module is self-contained. Run from inside its directory so relative config paths resolve correctly.

```bash
# LightGBM baseline
cd src/lightgbm_baseline && python train.py

# Neural network
cd src/neural_network && python train.py

# Student→Master pseudo-labeling pipeline
cd src/pseudo_labeling_pipeline && python train.py

# LightGBM + CatBoost ensemble
cd src/ensemble/lgbm_catboost && python train.py

# LightGBM + NN ensemble
cd src/ensemble/lgbm_nn && python train.py
```

## Acknowledgements

Techniques in this project were studied from public Kaggle write-ups and discussions.
See the [competition discussion page](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion)
for original sources and community solutions.
