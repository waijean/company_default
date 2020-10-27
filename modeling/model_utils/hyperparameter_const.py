RAW_DATA = {
    "n_estimators": 763,
    "min_samples_split": 20,
    "min_samples_leaf": 1,
    "max_features": "auto",
    "max_depth": 20,
    "bootstrap": True,
    "random_state": 42,
}

RATIO_DATA = {
    "n_estimators": 242,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "auto",
    "max_depth": 20,
    "bootstrap": False,
    "random_state": 42,
}

COMBINE_DATA = {
    # "n_estimators": 242,
    # "min_samples_split": 5,
    # "min_samples_leaf": 1,
    # "max_features": "auto",
    # "max_depth": 20,
    # "bootstrap": False,
    # best performance - 50
    "n_estimators": 857,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 60,
    "bootstrap": True,
    # best recall - 50
    # 'n_estimators': 289,
    # 'min_samples_split': 2,
    # 'min_samples_leaf': 2,
    # 'max_features': 'sqrt',
    # 'max_depth': 60,
    # 'bootstrap': False,
    "random_state": 42,
}
