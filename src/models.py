"""
Model training, hyperparameter tuning, and persistence module.
"""

import os
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODELS_DIR, RANDOM_STATE, CV_FOLDS


MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.joblib")


def get_models_and_params():
    """Return models and their hyperparameter grids for GridSearchCV."""
    return {
        "Multinomial NB": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
            },
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "params": {
                "C": [0.1, 1.0, 5.0, 10.0],
                "solver": ["lbfgs"],
            },
        },
        "Linear SVM": {
            "model": LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
            "params": {
                "C": [0.1, 0.5, 1.0, 5.0],
                "loss": ["hinge", "squared_hinge"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 30, 50],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        },
    }


def train_with_gridsearch(X_train, y_train, X_test, y_test):
    """Train all models with GridSearchCV and return results."""
    models_config = get_models_and_params()
    results = []
    
    print("=" * 60)
    print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 60)
    
    for name, config in models_config.items():
        print(f"\n{'-' * 50}")
        print(f"  Training: {name}")
        print(f"  Grid search over {len(config['params'])} parameter(s)...")
        
        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=CV_FOLDS,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
        
        results.append({
            "name": name,
            "model": best_model,
            "accuracy": acc,
            "f1_weighted": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "best_params": grid.best_params_,
            "preds": preds,
        })
        
        print(f"  Best params     : {grid.best_params_}")
        print(f"  Test Accuracy   : {acc * 100:.1f}%")
        print(f"  Weighted F1     : {f1 * 100:.1f}%")
        print(f"  Cross-Val Acc   : {cv_scores.mean() * 100:.1f}% (+/-{cv_scores.std() * 100:.1f}%)")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))
    
    # Find best model
    best = max(results, key=lambda r: r["f1_weighted"])
    print("=" * 60)
    print(f">> BEST MODEL: {best['name']} (F1={best['f1_weighted']*100:.1f}%, Acc={best['accuracy']*100:.1f}%)")
    print(f"   Best params: {best['best_params']}")
    print("=" * 60)
    
    return results, best


def save_model(model, vectorizer, preprocessor, metadata=None):
    """Save the trained model, vectorizer, and metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # For SVM models, wrap in CalibratedClassifierCV for probability support
    # (Already fitted, so we save as-is)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    meta = metadata or {}
    meta["preprocessor_config"] = {
        "remove_stopwords": preprocessor.remove_stopwords,
        "lemmatize": preprocessor.lemmatize,
    }
    joblib.dump(meta, METADATA_PATH)
    
    print(f"\n[OK] Model saved to: {MODELS_DIR}")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {VECTORIZER_PATH}")
    print(f"  -> {METADATA_PATH}")


def load_model():
    """Load a saved model, vectorizer, and metadata."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, METADATA_PATH]):
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    return model, vectorizer, metadata


def predict_genre(text, model, vectorizer, preprocessor):
    """Predict genre for a single text input."""
    cleaned = preprocessor.clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    
    # Try to get probabilities
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_labels = model.classes_
        probabilities = dict(zip(class_labels, probabilities))
    
    return prediction, probabilities
