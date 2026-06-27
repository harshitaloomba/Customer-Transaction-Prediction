# preprocess.py — LightGBM + NN Ensemble
# Creates base features shared by both pipelines.
# Model-specific transforms (rank vs RankGauss+scale) are applied in model.py.

import numpy as np
import pandas as pd


def create_base_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build shared base features: unseen-count, frequency encoding, row statistics.
    Does NOT apply rank transform or Gaussian normalisation here —
    those are applied per-model in model.py (prepare_lgbm_data / prepare_nn_data).

    Returns
    -------
    train_processed : pd.DataFrame
    test_processed  : pd.DataFrame
    """
    print("  [Preprocess] Creating base features...")
    features = [col for col in train_df.columns if 'var_' in col]

    lookup_sets = {col: set(train_df[col].unique()) for col in features}
    full_df = pd.concat([train_df[features], test_df[features]], sort=False)

    # Unseen-count
    unseen_cols_temp = []
    for col in features:
        tmp = col + '_unseen_temp'
        full_df[tmp] = (~full_df[col].isin(lookup_sets[col])).astype(int)
        unseen_cols_temp.append(tmp)
    full_df['unseen_count'] = full_df[unseen_cols_temp].sum(axis=1)
    full_df.drop(columns=unseen_cols_temp, inplace=True)

    # Frequency encoding
    for col in features:
        counts = full_df[col].value_counts()
        full_df[col + '_freq'] = full_df[col].map(counts)

    # Row-wise statistics (on raw values — rank applied later per model)
    full_df['sum']  = full_df[features].sum(axis=1)
    full_df['min']  = full_df[features].min(axis=1)
    full_df['max']  = full_df[features].max(axis=1)
    full_df['mean'] = full_df[features].mean(axis=1)
    full_df['std']  = full_df[features].std(axis=1)
    full_df['skew'] = full_df[features].skew(axis=1)
    full_df['kurt'] = full_df[features].kurtosis(axis=1)
    full_df['med']  = full_df[features].median(axis=1)

    train_processed = full_df[:len(train_df)]
    test_processed  = full_df[len(train_df):]

    return train_processed, test_processed
