# Results

## Model Performance (OOF AUROC)

| Model | OOF AUROC |
|---|---|
| LightGBM Baseline (10-fold CV) | 0.8947 |
| Neural Network (10-fold CV) | 0.8410 |
| LightGBM + CatBoost Ensemble | not logged — re-run required |
| LightGBM Master (pseudo-labeling only) | 0.8990 |
| **LGBM Student-Master Pipeline** | **0.9520** |
| LightGBM + NN Blend (50/50) | 0.8929 |

> **Note:** The PDF report's "0.95–0.96" range referred to the pseudo-labeling pipeline
> alone and not to the LightGBM + NN ensemble.

> **Note:** The LightGBM + CatBoost score was not captured in a final logged run.
> The experiment must be re-run to obtain a confirmed OOF AUROC value.

![Model Comparison](model_comparison.png)
