# preprocess.py — LightGBM Baseline
# Feature engineering: unseen-count, frequency encoding, rank transform, row statistics.
# Scaling is intentionally skipped — not required for LightGBM.

import pandas as pd
import numpy as np


def create_and_scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build enriched features from the raw 200 var_ columns.

    Steps
    -----
    1. Unseen-count  — flags how many test-row values were never seen in training.
    2. Frequency encoding — maps each value to its global occurrence count.
    3. Rank transform — normalises distributions and handles outliers.
    4. Row-wise statistics — sum, min, max, mean, std, skew, kurtosis, median
                             computed on the *ranked* features.

    Returns
    -------
    train_processed : pd.DataFrame
    test_processed  : pd.DataFrame
    all_features    : list[str]
    """
    print("Starting feature engineering...")
    features = [col for col in train_df.columns if 'var_' in col]

    # --- Unseen-count (Part 1): build per-column lookup sets from training data ---
    print("Creating value lookups from training data...")
    lookup_sets = {col: set(train_df[col].unique()) for col in features}

    # Combine train + test so all transformations are consistent
    full_df = pd.concat([train_df[features], test_df[features]], sort=False)

    # --- Unseen-count (Part 2) ---
    print("Calculating 'unseen_count' feature...")
    unseen_cols_temp = []
    for col in features:
        tmp = col + '_unseen_temp'
        full_df[tmp] = (~full_df[col].isin(lookup_sets[col])).astype(int)
        unseen_cols_temp.append(tmp)
    full_df['unseen_count'] = full_df[unseen_cols_temp].sum(axis=1)
    full_df.drop(columns=unseen_cols_temp, inplace=True)

    # --- Frequency encoding ---
    print("Calculating frequency encoding...")
    for col in features:
        freq_col = col + '_freq'
        counts = full_df[col].value_counts()
        full_df[freq_col] = full_df[col].map(counts)

    # --- Rank transform on the 200 original features ---
    print("Applying rank transformation...")
    full_df[features] = full_df[features].rank(method='average', na_option='bottom')

    # --- Row-wise statistics (on ranked features) ---
    print("Calculating row-wise statistics on ranked data...")
    full_df['sum']  = full_df[features].sum(axis=1)
    full_df['min']  = full_df[features].min(axis=1)
    full_df['max']  = full_df[features].max(axis=1)
    full_df['mean'] = full_df[features].mean(axis=1)
    full_df['std']  = full_df[features].std(axis=1)
    full_df['skew'] = full_df[features].skew(axis=1)
    full_df['kurt'] = full_df[features].kurtosis(axis=1)
    full_df['med']  = full_df[features].median(axis=1)

    print("Feature engineering complete. Scaling skipped (not needed for LightGBM).")

    train_processed = full_df[:len(train_df)]
    test_processed  = full_df[len(train_df):]
    all_features    = [c for c in train_processed.columns if c not in ['ID_code', 'target']]

    return train_processed[all_features], test_processed[all_features], all_features
