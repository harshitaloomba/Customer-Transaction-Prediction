# train.py — LightGBM + CatBoost Ensemble
# Trains both models with shared K-Fold splits and blends OOF predictions 50/50.
# Individual OOF scores logged per fold; final blend score reported at the end.

import gc
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from preprocess import create_and_scale_features

os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
warnings.filterwarnings('ignore')


def train_ensemble():
    # 1. Load data
    print("Loading data...")
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df  = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\nPlace train.csv / test.csv in data/ and update config.py paths.")
        return

    target           = train_df['target']
    train_to_process = train_df.drop(['ID_code', 'target'], axis=1)
    test_to_process  = test_df.drop(['ID_code'], axis=1)

    # 2. Shared feature engineering
    print("Preprocessing features...")
    train_X, test_X, _ = create_and_scale_features(train_to_process, test_to_process)

    # 3. OOF prediction arrays
    oof_lgbm = np.zeros(len(train_X))
    oof_cat  = np.zeros(len(train_X))

    # Class-imbalance weight for LightGBM
    neg_count, pos_count = np.bincount(target)
    scale_pos_weight     = neg_count / pos_count

    # 4. K-Fold loop — both models share the same folds
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    print(f"\nStarting training on {config.N_SPLITS} folds...")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_X, target)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")
        X_trn, X_val = train_X.iloc[trn_idx], train_X.iloc[val_idx]
        y_trn, y_val = target.iloc[trn_idx],  target.iloc[val_idx]

        # --- LightGBM ---
        print("  > Training LightGBM...")
        lgbm_clf = lgb.LGBMClassifier(**config.LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
        lgbm_clf.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        val_pred_lgbm   = lgbm_clf.predict_proba(X_val)[:, 1]
        oof_lgbm[val_idx] = val_pred_lgbm
        print(f"    LGBM fold AUC:     {roc_auc_score(y_val, val_pred_lgbm):.5f}")

        # --- CatBoost ---
        print("  > Training CatBoost...")
        cat_clf = CatBoostClassifier(
            **config.CATBOOST_PARAMS,
            scale_pos_weight=scale_pos_weight,
            verbose=False,
        )
        cat_clf.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True)
        val_pred_cat   = cat_clf.predict_proba(X_val)[:, 1]
        oof_cat[val_idx] = val_pred_cat
        print(f"    CatBoost fold AUC: {roc_auc_score(y_val, val_pred_cat):.5f}")

        del X_trn, X_val, y_trn, y_val, lgbm_clf, cat_clf
        gc.collect()

    # 5. Final scores
    auc_lgbm  = roc_auc_score(target, oof_lgbm)
    auc_cat   = roc_auc_score(target, oof_cat)
    blend_50  = roc_auc_score(target, 0.5 * oof_lgbm + 0.5 * oof_cat)

    print("\n==================================")
    print("FINAL RESULTS")
    print("==================================")
    print(f"LightGBM OOF AUROC:        {auc_lgbm:.5f}")
    print(f"CatBoost OOF AUROC:        {auc_cat:.5f}")
    print(f"Ensemble (50/50) AUROC:    {blend_50:.5f}")
    print("==================================")


if __name__ == "__main__":
    train_ensemble()
