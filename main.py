# Movie Genre Classification - Main Training Script
# Run this after fetching the dataset: python data/fetch_dataset.py

import os
import sys
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.utils import DATASET_PATH, RANDOM_STATE, TEST_SIZE, MAX_FEATURES, N_GRAM_RANGE, PLOTS_DIR
from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import train_with_gridsearch, save_model, load_model, predict_genre
from src.visualization import generate_all_plots

warnings.filterwarnings("ignore")


def load_and_explore(filepath):
    """Load dataset and print exploration stats."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    data = pd.read_csv(filepath)

    # Drop any NaN rows
    data = data.dropna(subset=["description", "genre"]).reset_index(drop=True)

    print(f"  Total samples : {len(data)}")
    print(f"  Columns       : {list(data.columns)}")
    print(f"  Genres found  : {data['genre'].nunique()}")
    print(f"\n  Samples per genre:")
    for genre, count in data["genre"].value_counts().sort_index().items():
        print(f"    {genre:<20s} {count}")
    print()
    return data


def run_pipeline(retrain=False):
    """Run the full ML pipeline."""

    # ── 1. Check dataset ────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print("[!] Dataset not found!")
        print(f"    Expected: {DATASET_PATH}")
        print(f"    Run: python data/fetch_dataset.py")
        sys.exit(1)

    # ── 2. Load & explore ───────────────────────────────────────────────
    data = load_and_explore(DATASET_PATH)

    # ── 3. Preprocess ───────────────────────────────────────────────────
    print("=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)

    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)

    # Keep original for word clouds
    data_original = data.copy()

    # Text stats before preprocessing
    stats_before = preprocessor.get_text_stats(data["description"])
    print(f"  Before preprocessing:")
    print(f"    Avg words    : {stats_before['avg_words']:.1f}")
    print(f"    Vocabulary   : {stats_before['total_vocab']}")

    # Apply preprocessing
    data["description"] = preprocessor.preprocess_series(data["description"])

    stats_after = preprocessor.get_text_stats(data["description"])
    print(f"  After preprocessing (lemmatization + stop words):")
    print(f"    Avg words    : {stats_after['avg_words']:.1f}")
    print(f"    Vocabulary   : {stats_after['total_vocab']}")
    print(f"  [OK] Preprocessing complete\n")

    # ── 4. Split ────────────────────────────────────────────────────────
    X = data["description"]
    y = data["genre"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── 5. Feature engineering ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    fe = FeatureEngineer(
        max_features=MAX_FEATURES,
        ngram_range=N_GRAM_RANGE,
        stop_words="english",
    )

    # Compare vectorizers
    comparison = fe.compare_vectorizers(X_train, X_test, y_train, y_test)
    print(f"  CountVectorizer : {comparison['count_vectorizer']['accuracy']*100:.1f}% "
          f"({comparison['count_vectorizer']['features']} features)")
    print(f"  TF-IDF (bigrams): {comparison['tfidf']['accuracy']*100:.1f}% "
          f"({comparison['tfidf']['features']} features)")
    print(f"  -> Using TF-IDF with {N_GRAM_RANGE} n-gram range\n")

    # Fit TF-IDF on training data
    X_train_vec = fe.fit_transform(X_train)
    X_test_vec = fe.transform(X_test)

    print(f"  Feature matrix : {X_train_vec.shape}")
    print(f"  Vocabulary size: {fe.get_vocab_size()}")

    # Top features per genre
    top_features = fe.get_top_features_per_class(X_train_vec, y_train, top_n=15)

    # ── 6. Model training with GridSearchCV ─────────────────────────────
    results, best = train_with_gridsearch(X_train_vec, y_train, X_test_vec, y_test)

    # ── 7. Save best model ──────────────────────────────────────────────
    save_model(
        model=best["model"],
        vectorizer=fe.tfidf,
        preprocessor=preprocessor,
        metadata={
            "best_model_name": best["name"],
            "accuracy": best["accuracy"],
            "f1_weighted": best["f1_weighted"],
            "best_params": best["best_params"],
            "genres": sorted(y.unique().tolist()),
        },
    )

    # ── 8. Generate all visualizations ──────────────────────────────────
    generate_all_plots(
        data=data_original,
        data_processed=data,
        results=results,
        best_result=best,
        X_train_vec=X_train_vec,
        y_train=y_train,
        y_test=y_test,
        top_features=top_features,
    )

    # ── 9. Interactive prediction ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENRE PREDICTOR -- Type a movie description (or 'quit' to exit)")
    print("=" * 60)

    while True:
        user_input = input("\n>> Enter movie description: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            print("[!] Please enter a description.")
            continue

        prediction, probs = predict_genre(user_input, best["model"], fe.tfidf, preprocessor)
        print(f">> Predicted Genre: {prediction}")

        if probs:
            print("   Confidence scores:")
            for genre, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
                bar = "#" * int(prob * 30)
                print(f"     {genre:<20s} {prob*100:5.1f}% {bar}")


if __name__ == "__main__":
    run_pipeline()
