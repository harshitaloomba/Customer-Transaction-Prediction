# train.py — LightGBM Baseline
# 10-fold stratified CV with early stopping. OOF AUROC: 0.8947.

import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from preprocess import create_and_scale_features

os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
warnings.filterwarnings('ignore')


def run_training():
    # 1. Load data
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df  = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\nPlace train.csv / test.csv in data/ and update config.py paths.")
        return

    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")

    # 2. Split labels / features
    target           = train_df['target']
    train_to_process = train_df.drop(['ID_code', 'target'], axis=1)
    test_to_process  = test_df.drop(['ID_code'], axis=1)

    # 3. Feature engineering
    train_features, test_features, all_features = create_and_scale_features(
        train_to_process, test_to_process
    )
    print(f"Total features: {len(all_features)}")

    # 4. Class-imbalance weight
    neg_count, pos_count   = np.bincount(target)
    scale_pos_weight       = neg_count / pos_count
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # 5. K-Fold training
    oof_preds = np.zeros(len(train_features))
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, target)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")
        X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        model = lgb.LGBMClassifier(**config.LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        val_preds          = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.4f}")

    # 6. Report
    overall_auc = roc_auc_score(target, oof_preds)
    print(f"\n--- Overall OOF AUC: {overall_auc:.4f} ---")
    print("Training complete.")


if __name__ == "__main__":
    run_training()
