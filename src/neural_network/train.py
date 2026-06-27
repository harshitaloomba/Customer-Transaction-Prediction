# train.py — Neural Network
# 10-fold stratified CV with early stopping + LR reduction. OOF AUROC: 0.8410.

import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from model import build_model
from preprocess import create_and_scale_features

os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
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

    # 3. Feature engineering + scaling
    train_scaled, test_scaled, all_features = create_and_scale_features(
        train_to_process, test_to_process
    )
    print(f"Total features: {len(all_features)}")

    # 4. Class weights for imbalanced data
    neg_count, pos_count = np.bincount(target)
    total = neg_count + pos_count
    class_weight = {
        0: (1 / neg_count) * (total / 2.0),
        1: (1 / pos_count) * (total / 2.0),
    }
    print(f"Class weights: {class_weight}")

    # 5. K-Fold training
    oof_preds = np.zeros(len(train_scaled))
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_scaled, target)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

        X_train, X_val = train_scaled[train_idx], train_scaled[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        model = build_model(X_train.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=5, restore_best_weights=True, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.1, patience=3, mode='max'
            ),
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        val_preds          = model.predict(X_val).ravel()
        oof_preds[val_idx] = val_preds
        print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds):.4f}")

    # 6. Report
    overall_auc = roc_auc_score(target, oof_preds)
    print(f"\n--- Overall OOF AUC: {overall_auc:.4f} ---")
    print("Training complete.")


if __name__ == "__main__":
    run_training()
