# config.py — LightGBM + Neural Network Ensemble

TRAIN_FILE  = '../../../data/train.csv'
TEST_FILE   = '../../../data/test.csv'

N_SPLITS    = 5
RANDOM_SEED = 42

# Shuffle augmentation
N_POS_AUG    = 4
N_NEG_AUG    = 1
N_PSEUDO_POS = 2700
N_PSEUDO_NEG = 2000

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

# Neural Network parameters
NN_EPOCHS     = 30
NN_BATCH_SIZE = 1024
NN_LR         = 0.001
