# config.py — LightGBM + CatBoost Ensemble

TRAIN_FILE  = '../../../data/train.csv'
TEST_FILE   = '../../../data/test.csv'

N_SPLITS    = 5
RANDOM_SEED = 42

# LightGBM parameters (tuned for this dataset)
LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.01,
    'num_leaves':       13,
    'max_depth':        -1,
    'min_data_in_leaf': 80,
    'bagging_freq':     5,
    'bagging_fraction': 0.4,
    'feature_fraction': 0.05,
    'boost_from_average': False,
    'verbosity':        -1,
    'n_jobs':           -1,
    'n_estimators':     100000,   # early stopping cuts this short
}

# CatBoost parameters
CATBOOST_PARAMS = {
    'loss_function':        'Logloss',
    'eval_metric':          'AUC',
    'learning_rate':        0.01,
    'iterations':           100000,
    'depth':                6,
    'l2_leaf_reg':          3,
    'random_seed':          42,
    'od_type':              'Iter',  # overfitting detector
    'od_wait':              200,
    'allow_writing_files':  False,
    'task_type':            'CPU',   # change to 'GPU' if available
}
