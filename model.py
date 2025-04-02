import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error
import joblib
import os

# Dummy-Daten generieren (Bitte durch reale Daten ersetzen)
data = {
    'age': [25, 30, 45, 50, 23, 35, 60, 28],
    'website_visits': [5, 10, 15, 20, 2, 8, 25, 30],
    'email_interactions': [1, 3, 4, 5, 0, 2, 5, 6],
    'purchased_before': [1, 0, 1, 1, 0, 0, 1, 1],  # 1=ja, 0=nein
    'lead_score': [80, 60, 90, 85, 40, 70, 95, 90]  # Zielwert, z.B. Score 0-100
}

df = pd.DataFrame(data)

# Features und Zielvariable definieren
X = df[['age', 'website_visits', 'email_interactions', 'purchased_before']]
y = df['lead_score']

# Splitte die Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelle definieren
models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# GridSearch für Hyperparameter-Tuning
param_grid = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10, 15]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    'LightGBM': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
}

# Modelle trainieren und evaluieren
best_model = None
best_score = float('inf')

for model_name, model in models.items():
    grid = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    
    # Bestes Modell finden
    score = mean_squared_error(y_test, grid.predict(X_test))
    print(f'{model_name} MSE: {score}')
    
    if score < best_score:
        best_score = score
        best_model = grid.best_estimator_

# Modell speichern
joblib.dump(best_model, 'lead_scorer_model.pkl')
