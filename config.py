from omegaconf import OmegaConf
config = {
    'paths': {
        'train': 'titanic/train.csv',
        'test': 'titanic/test.csv',
        'submission': 'titanic/submission.csv',
    },
    'features': {
        'drop': ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        'target': 'Survived',
    },
    'model': {
        'types': [
            'RandomForestClassifier',
            'LogisticRegression',
            'XGBClassifier',
            'HistGradientBoostingClassifier',
            'ExtraTreesClassifier'
        ],
        'params': {
            'RandomForestClassifier': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42,
            },
            'LogisticRegression': {
                'solver': 'liblinear',
                'random_state': 42,
            },
            'XGBClassifier': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            },
            'HistGradientBoostingClassifier': {
                'max_iter': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            },
            'ExtraTreesClassifier': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42,
            },
        }
    },
    'cross_validation': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42,
    }
}
py_config = OmegaConf.create(config)
print(config["model"]["types"], type(config["model"]["types"]))
py_config = OmegaConf.create(config)
print(py_config.model.get("types"), type(py_config.model.get("types")))
print(py_config.cross_validation.get('n_splits'), type(py_config.cross_validation.get('n_splits')))