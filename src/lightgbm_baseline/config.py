# config.py — LightGBM Baseline

# File paths (place train.csv and test.csv in data/ and update these)
TRAIN_FILE = '../../data/train.csv'
TEST_FILE  = '../../data/test.csv'

# Training hyperparameters
N_SPLITS    = 10
RANDOM_SEED = 42

# LightGBM parameters
LGBM_PARAMS = {
    'objective':      'binary',
    'metric':         'auc',
    'boosting_type':  'gbdt',
    'n_estimators':   2000,
    'learning_rate':  0.01,
    'num_leaves':     31,
    'max_depth':      -1,
    'seed':           42,
    'n_jobs':         -1,
    'verbose':        -1,
    'colsample_bytree': 0.6,
    'subsample':      0.8,
    'reg_alpha':      0.1,
    'reg_lambda':     0.1,
}
