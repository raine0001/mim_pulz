# Configuration settings for the adaptive AI interpretation project

# Data configuration
DATA_CONFIG = {
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "cross_validation_folds": 5,
}

# Model configuration
MODEL_CONFIG = {
    "models_to_try": [
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "neural_network",
    ],
    "hyperparameter_optimization": {
        "method": "optuna",  # or 'grid_search', 'random_search'
        "n_trials": 100,
        "timeout": 3600,  # seconds
    },
    "ensemble_methods": ["voting", "stacking", "blending"],
}

# Adaptive interpretation configuration
INTERPRETATION_CONFIG = {
    "methods": ["shap", "lime", "permutation_importance", "eli5"],
    "adaptation_threshold": 0.1,
    "drift_detection": {
        "method": "ks_test",  # or 'chi2_test', 'psi'
        "threshold": 0.05,
        "window_size": 100,
    },
    "feature_importance_stability": {"window_size": 5, "threshold": 0.2},
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "early_stopping": {
        "patience": 10,
        "monitor": "val_loss",
        "restore_best_weights": True,
    },
    "learning_rate_schedule": {
        "initial_lr": 0.001,
        "decay_factor": 0.5,
        "decay_patience": 5,
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "scaling": {
        "method": "standard",  # or 'minmax', 'robust'
        "features_to_scale": "numeric",
    },
    "encoding": {
        "categorical_method": "label",  # or 'onehot', 'target'
        "handle_unknown": "ignore",
    },
    "feature_selection": {
        "method": "recursive_elimination",  # or 'univariate', 'from_model'
        "n_features": "auto",
        "scoring": "f1_weighted",
    },
    "adaptive_features": {
        "rolling_window": 5,
        "lag_features": True,
        "interaction_features": True,
        "polynomial_degree": 2,
    },
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"],
    "cross_validation": {
        "method": "stratified_kfold",
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    "threshold_optimization": {
        "method": "youden_j",  # or 'f1_optimal', 'precision_recall'
        "cv_folds": 5,
    },
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_to_file": True,
    "log_file": "outputs/training.log",
}

# Output configuration
OUTPUT_CONFIG = {
    "model_save_path": "models/",
    "results_save_path": "outputs/",
    "plots_save_path": "outputs/plots/",
    "submissions_save_path": "outputs/submissions/",
    "save_format": "pickle",  # or 'joblib', 'h5'
    "compression": True,
}

# Experiment tracking configuration
EXPERIMENT_CONFIG = {
    "tracking_enabled": True,
    "tracking_system": "mlflow",  # or 'wandb', 'tensorboard'
    "experiment_name": "adaptive_ai_interpretation",
    "auto_log": True,
    "log_artifacts": True,
}
