import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer

from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

import xgboost as xgb

from config import py_config


""" 
Pipeline:
1. Load data
2. Preprocess data
    2.1. Split data into features and target
    2.2. Save "PassengerId" for submission(test set)
    2.3. BaseLine features: move to trash some noise features
    2.4. Split train data to train and validation sets with stratification / train_test_split 
    2.5. Impute missing values with SimpleImputer
    2.6. Scale numerical features with StandardScaler
    2.7. Encode cat features with one-hot encoding
3. Fit models and evaluate with cross-validation
4. Chose the best model and make pradictions on test set
5. Submit results on Kaggle
"""
#1 Load data
train = pd.read_csv(py_config.paths.train)
test = pd.read_csv(py_config.paths.test)

#2 preprocess data
y = train[py_config.features.target]
X = train.drop(columns=py_config.features.drop + [py_config.features.target])
test_X = test.drop(columns=py_config.features.drop)

cat_cols = ['Sex', 'Embarked']
num_cols = [c for c in X.columns if c not in cat_cols and c != 'Survived']

#2.5
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# --------- Some models -------------
MODEL_REGISTRY = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'XGBClassifier': xgb.XGBClassifier
}
# Combination model + param - instance
def build_models(py_config):
    models = {}
    names = py_config.model.types
    if names is None:
        raise ValueError('ModelType is not found in config')

    for name in names:
        cls = MODEL_REGISTRY[name]
        param_cfg = py_config.model.params.get(name, {})
        param = OmegaConf.to_container(param_cfg, resolve=True) # Перевели из Dict.conf -> dict
        models[name] = cls(**param)
    return models

models = build_models(py_config)

#----------- Cross-Validation -------------
cv = StratifiedKFold(n_splits=py_config.cross_validation.n_splits,
                     shuffle=py_config.cross_validation.shuffle,
                     random_state=py_config.cross_validation.random_state)

# Run model with cv and evaluate their accuracy
scores = {}
for name, model in models.items():
    pipe = Pipeline(steps=[('prep', preprocess), ('model', model)])
    cv_score = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    scores[name] = (cv_score.mean(), cv_score.std())
    print(f'{name}: {cv_score.mean():.4f} ± {cv_score.std():.4f}')

# ----------- Choose the best model ------------
best_model_name = max(scores, key=lambda k: scores[k][0])
print(f'Best model: {best_model_name} with accuracy {scores[best_model_name][0]:.4f}')

best_model = models[best_model_name]
best_pipe = Pipeline(steps=[('prep', preprocess), ('model', best_model)])
best_pipe.fit(X,y)

# Run the best model on test set
test_pred = best_pipe.predict(test_X)

#Do submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)
print('submission create')


