# preprocess.py — LightGBM + CatBoost Ensemble
# Shared feature engineering used by both models in the ensemble.

import numpy as np
import pandas as pd


def create_and_scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build enriched features for the LightGBM+CatBoost ensemble.
    CatBoost handles raw features well but consistent engineering improves both.

    Returns
    -------
    train_processed : pd.DataFrame
    test_processed  : pd.DataFrame
    all_features    : list[str]
    """
    print("  [Preprocess] Creating features...")
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

    # Rank transform
    full_df[features] = full_df[features].rank(method='average', na_option='bottom')

    # Row-wise statistics
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
    all_features    = list(train_processed.columns)

    return train_processed[all_features], test_processed[all_features], all_features
