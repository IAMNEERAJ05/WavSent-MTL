# config/config.py
# WavSent-MTL — Single Source of Truth
# DO NOT hardcode any value in src/ files
# Import everywhere:
# from config.config import CONFIG

import os as _os

# Derive project root from this file's location — works locally (D:/WavSent-MTL)
# and on Kaggle (/kaggle/working/WavSent-MTL) without any changes.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace('\\', '/')

CONFIG = {

    # ─────────────────────────────────────
    # PATHS  (absolute, environment-agnostic)
    # ─────────────────────────────────────
    'project_root':          _ROOT + '/',
    'raw_data_dir':          _ROOT + '/data/raw/',
    'finbert_output_dir':    _ROOT + '/data/finbert_outputs/',
    'kotekar_processed_dir': _ROOT + '/data/processed/kotekar/',
    'kaggle_processed_dir':  _ROOT + '/data/processed/kaggle/',
    'results_dir':           _ROOT + '/results/',
    'figures_dir':           _ROOT + '/results/figures/',
    'tables_dir':            _ROOT + '/results/tables/',
    'models_dir':            _ROOT + '/results/saved_models/',
    'logs_dir':              _ROOT + '/logs/training_logs/',
    'ablation_dir':          _ROOT + '/ablation/results/',

    # ─────────────────────────────────────
    # DATA
    # ─────────────────────────────────────
    'ticker':                '^NSEI',
    'price_start':           '2017-01-01',
    'price_end':             '2024-05-31',

    'kotekar_start':         '2020-01-01',
    'kotekar_end':           '2024-05-31',

    'kaggle1_start':         '2017-01-01',
    'kaggle1_end':           '2021-04-30',
    'kaggle2_start':         '2022-01-01',
    'kaggle2_end':           '2024-05-31',

    'gap_start':             '2021-05-01',
    'gap_end':               '2021-12-31',

    # ─────────────────────────────────────
    # FINBERT OUTPUT COLUMNS
    # ─────────────────────────────────────
    'kotekar_date_col':      'date',
    'kotekar_polarity_col':  'polarity_mean',

    'kaggle_date_col':       'date',
    'kaggle_mean_col':       'polarity_mean',
    'kaggle_max_col':        'polarity_max',

    # ─────────────────────────────────────
    # MISSING SENTIMENT FILL VALUES
    # ─────────────────────────────────────
    'missing_polarity_mean': 0.0,
    'missing_polarity_max':  0.0,

    # ─────────────────────────────────────
    # WAVELET
    # ─────────────────────────────────────
    'wavelet':               'coif3',
    'wavelet_level':         1,
    'wavelet_mode':          'soft',
    'ohlcv_cols':            [
        'Open', 'High', 'Low',
        'Close', 'Volume'],

    # ─────────────────────────────────────
    # TECHNICAL INDICATOR PARAMETERS
    # ─────────────────────────────────────
    'rsi_period':            14,
    'macd_fast':             12,
    'macd_slow':             26,
    'bb_period':             20,
    'bb_std':                2,
    'roc_period':            5,
    'ema_period':            9,
    'atr_period':            14,
    'stoch_period':          14,
    'williams_period':       14,
    'cci_period':            20,

    # ─────────────────────────────────────
    # CANDIDATE FEATURES (all 15)
    # ─────────────────────────────────────
    'candidate_features': [
        'Close_d', 'Open_d', 'High_d',
        'Low_d', 'Volume_d',
        'RSI_14', 'MACD', 'BB_width',
        'ROC_5', 'EMA_9', 'ATR_14',
        'OBV', 'STOCH_K',
        'WILLIAMS_R', 'CCI_20'
    ],

    # ─────────────────────────────────────
    # FEATURE SELECTION
    # ─────────────────────────────────────
    'mi_top_k':              10,
    'shap_top_k':            7,
    'feature_selection_seed': 42,
    'feature_selection_runs': 10,
    'feature_selection_units': 32,

    # ─────────────────────────────────────
    # BEST REPRESENTATION
    # SET THIS MANUALLY AFTER B vs C COMPARISON
    # Options: 'denoised_ohlcv' or
    #          'denoised_technicals'
    # ─────────────────────────────────────
    'BEST_REPR': 'denoised_technicals',

    # ─────────────────────────────────────
    # DUAL REPRESENTATION (OPTIONAL)
    # ─────────────────────────────────────
    'DUAL_REPR': False,

    # ─────────────────────────────────────
    # DATA SPLIT
    # ─────────────────────────────────────
    'train_ratio':           0.70,
    'val_ratio':             0.15,
    'test_ratio':            0.15,
    'shuffle':               False,
    'window_size':           5,
    'warmup_days':           26,

    # ─────────────────────────────────────
    # CLASS IMBALANCE
    # ─────────────────────────────────────
    'imbalance_threshold':   1.5,

    # ─────────────────────────────────────
    # BEST HYPERPARAMETERS
    # UPDATED AFTER NOTEBOOK 05
    # ─────────────────────────────────────
    'best_params': {
        'tkan': {
            'hidden_size':   64,
            'dropout':       0.1,
            'learning_rate': 1e-3,
            'batch_size':    32,
        },
        'lstm': {
            'hidden_size':   64,
            'dropout':       0.1,
            'learning_rate': 1e-3,
            'batch_size':    32,
            'num_layers':    2,
        },
        'gru': {
            'hidden_size':   128,
            'dropout':       0.3,
            'learning_rate': 1e-3,
            'batch_size':    32,
            'num_layers':    1,
        },
        'tcn': {
            'hidden_size':   64,
            'dropout':       0.1,
            'learning_rate': 1e-3,
            'batch_size':    32,
            'kernel_size':   2,
            'num_levels':    2,
        },
    },

    # ─────────────────────────────────────
    # HYPERPARAMETER SEARCH SPACES
    # ─────────────────────────────────────
    'search_spaces': {
        'common': {
            'hidden_size':   [32, 64, 128],
            'dropout':       [0.1, 0.2, 0.3],
            'learning_rate': [1e-3, 5e-4, 1e-4],
            'batch_size':    [16, 32, 64],
        },
        'lstm': {'num_layers': [1, 2]},
        'gru':  {'num_layers': [1, 2]},
        'tcn':  {
            'kernel_size':   [2, 3],
            'num_levels':    [2, 3],
        },
    },
    'n_search_trials':       40,

    # ─────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────
    'optimizer':             'adam',
    'weight_decay':          1e-4,
    'max_epochs':              150,
    'early_stopping_patience': 35,
    'early_stopping_monitor':  'val_binary_accuracy',
    'restore_best_weights':    True,
    'lr_reduce_factor':        0.5,
    'lr_reduce_patience':      10,
    'lr_reduce_monitor':       'val_loss',
    'lr_min':                  1e-6,
    'grad_clip_norm':          1.0,
    'n_runs':                  30,

    # ─────────────────────────────────────
    # LOSS
    # ─────────────────────────────────────
    'loss_type':             'uncertainty',
    'fixed_mse_weight':      0.3,
    'fixed_bce_weight':      0.7,
    'log_sigma_init':        0.0,

    # ─────────────────────────────────────
    # PSO
    # ─────────────────────────────────────
    'pso_n_particles':       20,
    'pso_iterations':        50,
    'pso_c1':                0.5,
    'pso_c2':                0.3,
    'pso_w':                 0.9,
    'pso_models':            [
        'tkan', 'lstm', 'gru', 'tcn'],

    # ─────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────
    'risk_free_rate':        0.06,
    'trading_strategy':      'long_only',
    'granger_max_lag':       5,
    'wilcoxon_config_a':     'A',
    'wilcoxon_config_b':     'G',

    # ─────────────────────────────────────
    # ABLATION CONFIGS
    # ─────────────────────────────────────
    'ablation_configs': {
        'A': {
            'input_type':  'returns_sentiment',
            'model':       'tkan',
            'n_runs':      30,
            'description': 'Returns + polarity_mean,'
                           ' TKAN, MTL baseline'
        },
        'B': {
            'input_type':  'denoised_ohlcv',
            'model':       'tkan',
            'n_runs':      30,
            'description': 'Denoised OHLCV + '
                           'polarity_mean, TKAN'
        },
        'C': {
            'input_type':  'denoised_technicals',
            'model':       'tkan',
            'n_runs':      30,
            'description': 'Denoised technicals + '
                           'polarity_mean, TKAN'
        },
        'D': {
            'input_type':  'BEST_REPR',
            'model':       'lstm',
            'n_runs':      30,
            'description': 'Best repr, LSTM'
        },
        'E': {
            'input_type':  'BEST_REPR',
            'model':       'gru',
            'n_runs':      30,
            'description': 'Best repr, GRU'
        },
        'F': {
            'input_type':  'BEST_REPR',
            'model':       'tcn',
            'n_runs':      30,
            'description': 'Best repr, TCN'
        },
        'G': {
            'input_type':  'BEST_REPR',
            'model':       'ensemble',
            'n_runs':      1,
            'description': 'PSO ensemble of '
                           'TKAN+LSTM+GRU+TCN'
        },
    },

    # ─────────────────────────────────────
    # RESULTS COLUMNS
    # ─────────────────────────────────────
    'results_columns': [
        'config', 'model', 'seed', 'run',
        'dataset',
        'accuracy', 'balanced_accuracy',
        'auc', 'precision', 'recall', 'f1',
        'rmse', 'mae', 'r2',
        'val_accuracy'
    ],
}
