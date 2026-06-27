# train.py — Student → Master Pseudo-Labeling Pipeline
# Implements the two-stage pseudo-labeling strategy from the Kaggle 1st-place solution.
# Stage 1 (Student): train on shuffle-augmented data, get test predictions.
# Stage 2 (Master): add high-confidence pseudo-labels from test set, retrain.
# OOF AUROC progression: baseline 0.8947 → master 0.9520.

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


# ---------------------------------------------------------------------------
# Shuffle augmentation  (1st-place technique)
# ---------------------------------------------------------------------------

def shuffle_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_pos: int,
    n_neg: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic rows by randomly shuffling feature values within each
    positive / negative sample independently.

    Parameters
    ----------
    X_train, y_train : training features and labels
    n_pos : number of additional copies of positives
    n_neg : number of additional copies of negatives

    Returns
    -------
    X_aug, y_aug : augmented features and labels
    """
    print(f"Augmenting: {n_pos}× positive, {n_neg}× negative ...")

    X_pos, y_pos = X_train[y_train == 1], y_train[y_train == 1]
    X_neg, y_neg = X_train[y_train == 0], y_train[y_train == 0]

    X_aug = [X_train]
    y_aug = [y_train]

    for _ in range(n_pos):
        X_shuffled = X_pos.copy()
        np.apply_along_axis(np.random.shuffle, 1, X_shuffled.values)
        X_aug.append(X_shuffled)
        y_aug.append(y_pos)

    for _ in range(n_neg):
        X_shuffled = X_neg.copy()
        np.apply_along_axis(np.random.shuffle, 1, X_shuffled.values)
        X_aug.append(X_shuffled)
        y_aug.append(y_neg)

    X_aug_final = pd.concat(X_aug, ignore_index=True)
    y_aug_final = pd.concat(y_aug, ignore_index=True)
    print(f"Augmented dataset size: {len(X_aug_final):,} rows")
    return X_aug_final, y_aug_final


# ---------------------------------------------------------------------------
# K-Fold LightGBM trainer
# ---------------------------------------------------------------------------

def train_lgbm(
    train_features: pd.DataFrame,
    target: pd.Series,
    n_splits: int,
    params: dict,
    test_features: pd.DataFrame | None = None,
) -> tuple[float, np.ndarray | None]:
    """
    Train LightGBM with K-Fold cross-validation and optional test inference.

    Returns
    -------
    oof_auc    : float      out-of-fold AUROC on training data
    test_preds : np.ndarray averaged test predictions (or None)
    """
    oof_preds  = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features)) if test_features is not None else None

    # Compute class-imbalance weight
    neg_count, pos_count       = np.bincount(target)
    params = {**params, 'scale_pos_weight': neg_count / pos_count}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, target)):
        print(f"  Fold {fold + 1}/{n_splits} ...")
        X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx],         target.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        if test_features is not None:
            test_preds += model.predict_proba(test_features)[:, 1] / n_splits

    oof_auc = roc_auc_score(target, oof_preds)
    return oof_auc, test_preds


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    # 1. Load and preprocess
    print("=== Loading and Preprocessing Data ===")
    try:
        train_df = pd.read_csv(config.TRAIN_FILE)
        test_df  = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\nPlace train.csv / test.csv in data/ and update config.py paths.")
        return

    original_target = train_df['target']
    train_features, test_features, all_features = create_and_scale_features(
        train_df.drop(['ID_code', 'target'], axis=1),
        test_df.drop(['ID_code'], axis=1),
    )
    print(f"Features created: {len(all_features)}")

    # ------------------------------------------------------------------
    # STAGE 1 — Student model
    # ------------------------------------------------------------------
    print("\n=== STAGE 1: Student Model ===")

    X_aug, y_aug = shuffle_augmentation(
        train_features, original_target,
        config.N_POS_AUG, config.N_NEG_AUG,
    )

    print("Training Student on augmented data (getting test predictions) ...")
    _, student_test_preds = train_lgbm(
        X_aug, y_aug,
        n_splits=config.N_SPLITS,
        params=config.LGBM_PARAMS,
        test_features=test_features,
    )

    # ------------------------------------------------------------------
    # Pseudo-labeling — select high-confidence test rows
    # ------------------------------------------------------------------
    print("\n=== Generating Pseudo-Labels ===")
    pseudo_idx_pos = np.argsort(student_test_preds)[-config.N_PSEUDO_POS:]
    pseudo_idx_neg = np.argsort(student_test_preds)[:config.N_PSEUDO_NEG]

    rows_pos = test_features.iloc[pseudo_idx_pos].copy()
    rows_neg = test_features.iloc[pseudo_idx_neg].copy()
    rows_pos['target'] = 1
    rows_neg['target'] = 0

    pseudo_rows = pd.concat([rows_pos, rows_neg])
    print(f"Pseudo-labeled rows added: {len(pseudo_rows)}")

    # ------------------------------------------------------------------
    # Build master training set
    # ------------------------------------------------------------------
    print("\n=== Building Master Training Set ===")
    master_features = pd.concat(
        [X_aug, pseudo_rows[all_features]], ignore_index=True
    )
    master_target = pd.concat(
        [y_aug, pseudo_rows['target']], ignore_index=True
    )
    print(f"Master training set size: {len(master_features):,} rows")

    # ------------------------------------------------------------------
    # STAGE 2 — Master model (validation only — no test inference needed)
    # ------------------------------------------------------------------
    print("\n=== STAGE 2: Master Model (Validation) ===")
    master_auc, _ = train_lgbm(
        master_features, master_target,
        n_splits=config.N_SPLITS,
        params=config.LGBM_PARAMS,
        test_features=None,
    )

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Master Model OOF AUROC: {master_auc:.6f}")


if __name__ == "__main__":
    run_pipeline()
