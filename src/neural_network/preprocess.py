# preprocess.py — Neural Network
# Feature engineering + StandardScaler normalisation (mandatory for NNs).
# Applies: unseen-count, frequency encoding, rank transform, row statistics, scaling.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_and_scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build enriched features and scale them for neural network input.

    Steps
    -----
    1. Unseen-count  — count how many values in each row were unseen in training.
    2. Frequency encoding — global occurrence count per value.
    3. Rank transform — normalises distributions, handles outliers.
    4. Row-wise statistics — computed on ranked features.
    5. StandardScaler — zero-mean, unit-variance scaling (required for NNs).

    Returns
    -------
    train_scaled : np.ndarray  shape (n_train, n_features)
    test_scaled  : np.ndarray  shape (n_test,  n_features)
    all_features : list[str]
    """
    print("Starting feature engineering...")
    features = [col for col in train_df.columns if 'var_' in col]

    # 1. Unseen-count
    print("Creating value lookups from training data...")
    lookup_sets = {col: set(train_df[col].unique()) for col in features}
    full_df = pd.concat([train_df[features], test_df[features]], sort=False)

    print("Calculating 'unseen_count' feature...")
    unseen_cols_temp = []
    for col in features:
        tmp = col + '_unseen_temp'
        full_df[tmp] = (~full_df[col].isin(lookup_sets[col])).astype(int)
        unseen_cols_temp.append(tmp)
    full_df['unseen_count'] = full_df[unseen_cols_temp].sum(axis=1)
    full_df.drop(columns=unseen_cols_temp, inplace=True)

    # 2. Frequency encoding
    print("Calculating frequency encoding...")
    for col in features:
        counts = full_df[col].value_counts()
        full_df[col + '_freq'] = full_df[col].map(counts)

    # 3. Rank transform
    print("Applying rank transformation...")
    full_df[features] = full_df[features].rank(method='average', na_option='bottom')

    # 4. Row-wise statistics (on ranked features)
    print("Calculating row-wise statistics on ranked data...")
    full_df['sum']  = full_df[features].sum(axis=1)
    full_df['min']  = full_df[features].min(axis=1)
    full_df['max']  = full_df[features].max(axis=1)
    full_df['mean'] = full_df[features].mean(axis=1)
    full_df['std']  = full_df[features].std(axis=1)
    full_df['skew'] = full_df[features].skew(axis=1)
    full_df['kurt'] = full_df[features].kurtosis(axis=1)
    full_df['med']  = full_df[features].median(axis=1)

    print("Feature engineering complete.")

    train_processed = full_df[:len(train_df)]
    test_processed  = full_df[len(train_df):]
    all_features    = [c for c in train_processed.columns if c not in ['ID_code', 'target']]

    # 5. StandardScaler
    print(f"Scaling {len(all_features)} features...")
    scaler      = StandardScaler()
    train_scaled = scaler.fit_transform(train_processed[all_features])
    test_scaled  = scaler.transform(test_processed[all_features])
    print("Scaling complete.")

    return train_scaled, test_scaled, all_features
