# model.py — LightGBM + NN Ensemble
# Contains:
#   - prepare_lgbm_data()   : rank transform for LightGBM
#   - prepare_nn_data()     : RankGauss + StandardScaler for the neural network
#   - shuffle_augmentation(): 1st-place shuffle trick
#   - build_nn()            : 3-hidden-layer MLP with BatchNorm + Dropout
#   - train_fold()          : single-fold trainer for either model type
#   - run_pipeline_for_model(): full Student→Pseudo-Label→Master loop per model

import gc
import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import config
from preprocess import create_base_features

os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def prepare_lgbm_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply average-rank transform to the 200 var_ features for LightGBM."""
    print("  [Data] Applying rank transform for LightGBM...")
    features = [c for c in train_df.columns if 'var_' in c]
    X_train, X_test = train_df.copy(), test_df.copy()
    full = pd.concat([X_train[features], X_test[features]])
    full = full.rank(method='average', na_option='bottom')
    X_train[features] = full.iloc[:len(X_train)]
    X_test[features]  = full.iloc[len(X_train):]
    return X_train, X_test


def prepare_nn_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply RankGauss (QuantileTransformer → normal) + StandardScaler for the NN."""
    print("  [Data] Applying RankGauss + StandardScaler for Neural Network...")
    features = [c for c in train_df.columns if 'var_' in c]
    X_train, X_test = train_df.copy(), test_df.copy()

    qt = QuantileTransformer(output_distribution='normal', random_state=config.RANDOM_SEED)
    full_features = pd.concat([X_train[features], X_test[features]])
    full_qt = qt.fit_transform(full_features)
    X_train[features] = full_qt[:len(X_train)]
    X_test[features]  = full_qt[len(X_train):]

    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train, X_test


# ---------------------------------------------------------------------------
# 2. Shuffle augmentation
# ---------------------------------------------------------------------------

def shuffle_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_pos: int,
    n_neg: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Duplicate rows with shuffled feature values to expand the training set."""
    X_pos, y_pos = X_train[y_train == 1], y_train[y_train == 1]
    X_neg, y_neg = X_train[y_train == 0], y_train[y_train == 0]

    X_aug, y_aug = [X_train], [y_train]
    for _ in range(n_pos):
        X_s = X_pos.copy()
        np.apply_along_axis(np.random.shuffle, 1, X_s.values)
        X_aug.append(X_s); y_aug.append(y_pos)
    for _ in range(n_neg):
        X_s = X_neg.copy()
        np.apply_along_axis(np.random.shuffle, 1, X_s.values)
        X_aug.append(X_s); y_aug.append(y_neg)

    return pd.concat(X_aug, ignore_index=True), pd.concat(y_aug, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Neural network architecture
# ---------------------------------------------------------------------------

def build_nn(input_shape: int) -> keras.Sequential:
    """
    3-hidden-layer MLP.
    Dense(512) → BN → ReLU → Dropout(0.3)
    Dense(256) → BN → ReLU → Dropout(0.3)
    Dense(128) → BN → ReLU → Dropout(0.3)
    Dense(1)   → Sigmoid
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),

        keras.layers.Dense(512),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(config.NN_LR),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Single-fold trainer
# ---------------------------------------------------------------------------

def train_fold(
    model_type: str,
    X_trn: pd.DataFrame, y_trn: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one fold for the given model type; return (val_preds, test_preds)."""
    if model_type == 'LGBM':
        neg, pos = np.bincount(y_trn)
        p = {**config.LGBM_PARAMS, 'scale_pos_weight': neg / pos}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        val_p  = model.predict_proba(X_val)[:, 1]
        test_p = model.predict_proba(X_test)[:, 1]

    elif model_type == 'NN':
        neg, pos = np.bincount(y_trn)
        total = neg + pos
        cw = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
        model = build_nn(X_trn.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=4, mode='max', restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=2, mode='max'
            ),
        ]
        model.fit(
            X_trn, y_trn,
            validation_data=(X_val, y_val),
            epochs=config.NN_EPOCHS,
            batch_size=config.NN_BATCH_SIZE,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0,
        )
        val_p  = model.predict(X_val,  verbose=0).ravel()
        test_p = model.predict(X_test, verbose=0).ravel()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return val_p, test_p


# ---------------------------------------------------------------------------
# 5. Student → Pseudo-Label → Master pipeline (per model)
# ---------------------------------------------------------------------------

def run_pipeline_for_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Run the full two-stage pipeline for one model type.

    Stage 1 — Student: train on augmented folds, collect test predictions.
    Pseudo-labeling: pick top-N_PSEUDO_POS and bottom-N_PSEUDO_NEG test rows.
    Stage 2 — Master: add pseudo rows + augment, retrain, return OOF preds.

    Returns
    -------
    oof_preds_master : np.ndarray  OOF predictions from the master model
    """
    print(f"\n=== PIPELINE: {model_name} ===")
    oof_student = np.zeros(len(X))
    test_student = np.zeros(len(X_test))

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    # --- Stage 1: Student ---
    print(f"  [{model_name}] Training Student...")
    for fold, (idx_t, idx_v) in enumerate(skf.split(X, y)):
        X_trn, X_val = X.iloc[idx_t], X.iloc[idx_v]
        y_trn, y_val = y.iloc[idx_t], y.iloc[idx_v]
        X_trn_aug, y_trn_aug = shuffle_augmentation(X_trn, y_trn, config.N_POS_AUG, config.N_NEG_AUG)
        val_p, test_p = train_fold(model_name, X_trn_aug, y_trn_aug, X_val, y_val, X_test)
        oof_student[idx_v]  = val_p
        test_student       += test_p / config.N_SPLITS
        print(f"    Fold {fold + 1} Student AUC: {roc_auc_score(y_val, val_p):.5f}")

    print(f"  >>> {model_name} Student OOF AUC: {roc_auc_score(y, oof_student):.6f}")

    # --- Pseudo-labeling ---
    print(f"  [{model_name}] Generating pseudo-labels...")
    idx_pos = np.argsort(test_student)[-config.N_PSEUDO_POS:]
    idx_neg = np.argsort(test_student)[:config.N_PSEUDO_NEG]
    rows_pos = X_test.iloc[idx_pos].copy(); rows_pos['target'] = 1
    rows_neg = X_test.iloc[idx_neg].copy(); rows_neg['target'] = 0
    X_pseudo = pd.concat([rows_pos, rows_neg]).drop('target', axis=1)
    y_pseudo = pd.concat([rows_pos, rows_neg])['target']

    # --- Stage 2: Master ---
    print(f"  [{model_name}] Training Master...")
    oof_master = np.zeros(len(X))
    for fold, (idx_t, idx_v) in enumerate(skf.split(X, y)):
        X_trn, X_val = X.iloc[idx_t], X.iloc[idx_v]
        y_trn, y_val = y.iloc[idx_t], y.iloc[idx_v]
        X_trn_combined = pd.concat([X_trn, X_pseudo], ignore_index=True)
        y_trn_combined = pd.concat([y_trn, y_pseudo], ignore_index=True)
        X_trn_aug, y_trn_aug = shuffle_augmentation(
            X_trn_combined, y_trn_combined, config.N_POS_AUG, config.N_NEG_AUG
        )
        val_p, _ = train_fold(model_name, X_trn_aug, y_trn_aug, X_val, y_val, X_test)
        oof_master[idx_v] = val_p
        print(f"    Fold {fold + 1} Master AUC: {roc_auc_score(y_val, val_p):.5f}")

    print(f"  >>> {model_name} Master OOF AUC: {roc_auc_score(y, oof_master):.6f}")
    return oof_master
