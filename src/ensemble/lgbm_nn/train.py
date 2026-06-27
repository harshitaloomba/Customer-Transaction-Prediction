# train.py — LightGBM + NN Ensemble
# Runs the Student→Master pipeline independently for LGBM and NN,
# then blends OOF predictions 50/50. OOF AUROC: 0.8929.
#
# Note: blending LGBM (0.899) with NN (0.841) hurt the overall score vs
# LGBM alone — structural diversity does not guarantee ensemble gains.

import gc
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from preprocess import create_base_features
from model import prepare_lgbm_data, prepare_nn_data, run_pipeline_for_model

os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
warnings.filterwarnings('ignore')


def run_training():
    # 1. Load data
    print("--- Loading Data ---")
    try:
        train = pd.read_csv(config.TRAIN_FILE)
        test  = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\nPlace train.csv / test.csv in data/ and update config.py paths.")
        return

    target = train['target']
    train_raw = train.drop(['ID_code', 'target'], axis=1)
    test_raw  = test.drop(['ID_code'], axis=1)

    # 2. Shared base features (no model-specific transforms yet)
    train_base, test_base = create_base_features(train_raw, test_raw)

    # 3. LGBM pipeline (rank transform applied inside prepare_lgbm_data)
    X_lgbm, X_test_lgbm = prepare_lgbm_data(train_base.copy(), test_base.copy())
    lgbm_oof = run_pipeline_for_model("LGBM", X_lgbm, target, X_test_lgbm)
    del X_lgbm, X_test_lgbm
    gc.collect()

    # 4. NN pipeline (RankGauss + scaling applied inside prepare_nn_data)
    X_nn, X_test_nn = prepare_nn_data(train_base.copy(), test_base.copy())
    nn_oof = run_pipeline_for_model("NN", X_nn, target, X_test_nn)
    del X_nn, X_test_nn
    gc.collect()

    # 5. Final blend and results
    blend_oof   = 0.5 * lgbm_oof + 0.5 * nn_oof
    blend_score = roc_auc_score(target, blend_oof)

    print("\n=== FINAL RESULTS ===")
    print(f"LightGBM Master OOF AUROC:    {roc_auc_score(target, lgbm_oof):.6f}")
    print(f"Neural Network Master AUROC:  {roc_auc_score(target, nn_oof):.6f}")
    print(f"---------------------------------------")
    print(f"Blended (50/50) AUROC:        {blend_score:.6f}")
    print(f"---------------------------------------")


if __name__ == "__main__":
    run_training()
