"""
Visualization module - generates 7 different plots for analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from wordcloud import WordCloud

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import PLOTS_DIR

# Style settings
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLOR_PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
]


def plot_genre_distribution(data):
    """1. Bar chart of genre distribution in the dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts = data["genre"].value_counts().sort_index()
    colors = COLOR_PALETTE[:len(genre_counts)]
    
    bars = ax.bar(range(len(genre_counts)), genre_counts.values, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.7)
    
    for bar, val in zip(bars, genre_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", fontweight="bold", fontsize=11)
    
    ax.set_xticks(range(len(genre_counts)))
    ax.set_xticklabels(genre_counts.index, rotation=30, ha="right", fontsize=10)
    ax.set_title("Genre Distribution in Dataset", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, max(genre_counts.values) * 1.15)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "genre_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_word_clouds(data, genres=None, max_words=100):
    """2. Word clouds for top genres."""
    if genres is None:
        genres = sorted(data["genre"].unique())[:4]  # Top 4
    
    n = len(genres)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, genre in enumerate(genres):
        r, c = idx // cols, idx % cols
        text = " ".join(data[data["genre"] == genre]["description"].astype(str))
        
        wc = WordCloud(
            width=600, height=300,
            max_words=max_words,
            background_color="white",
            colormap="viridis",
            contour_width=1,
            contour_color="#333333",
        ).generate(text if text.strip() else "empty")
        
        axes[r, c].imshow(wc, interpolation="bilinear")
        axes[r, c].set_title(f"{genre}", fontsize=14, fontweight="bold")
        axes[r, c].axis("off")
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    
    plt.suptitle("Word Clouds by Genre", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "word_clouds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_model_comparison(results):
    """3. Grouped bar chart comparing model accuracy and F1."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    f1_scores = [r["f1_weighted"] * 100 for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",
                   color="#2196F3", edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + width/2, f1_scores, width, label="F1-Score (Weighted)",
                   color="#FF5722", edgecolor="white", linewidth=1)
    
    for bar, val in zip(bars1, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
    for bar, val in zip(bars2, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, max(max(accuracies), max(f1_scores)) * 1.15)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_confusion_matrix(y_true, y_pred, model_name, labels):
    """4. Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Genre", fontsize=12)
    ax.set_ylabel("Actual Genre", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_learning_curve(model, X_train, y_train, model_name):
    """5. Learning curve showing train/validation accuracy vs training size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    
    train_mean = train_scores.mean(axis=1) * 100
    train_std = train_scores.std(axis=1) * 100
    val_mean = val_scores.mean(axis=1) * 100
    val_std = val_scores.std(axis=1) * 100
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2196F3")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#FF5722")
    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", linewidth=2, label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="#FF5722", linewidth=2, label="Validation Score")
    
    ax.set_title(f"Learning Curve - {model_name}", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "learning_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_top_features(top_features, n_genres=4):
    """6. Horizontal bar chart of top TF-IDF features per genre."""
    genres = list(top_features.keys())[:n_genres]
    n = len(genres)
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, genre in enumerate(genres):
        r, c = idx // cols, idx % cols
        features = top_features[genre]
        words = [f[0] for f in features][:10][::-1]
        scores = [f[1] for f in features][:10][::-1]
        
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        axes[r, c].barh(words, scores, color=color, edgecolor="white", height=0.6)
        axes[r, c].set_title(f"{genre}", fontsize=13, fontweight="bold")
        axes[r, c].set_xlabel("TF-IDF Score", fontsize=10)
    
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    
    plt.suptitle("Top TF-IDF Features by Genre", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "top_features.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def plot_text_length_distribution(data):
    """7. Box plot of text length distribution by genre."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_copy = data.copy()
    data_copy["word_count"] = data_copy["description"].apply(lambda x: len(str(x).split()))
    
    genres_sorted = sorted(data_copy["genre"].unique())
    box_data = [data_copy[data_copy["genre"] == g]["word_count"].values for g in genres_sorted]
    
    bp = ax.boxplot(box_data, labels=genres_sorted, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    
    for patch, color in zip(bp["boxes"], COLOR_PALETTE[:len(genres_sorted)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title("Text Length Distribution by Genre", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Word Count", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "text_length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Saved: {path}")
    return path


def generate_all_plots(data, data_processed, results, best_result,
                       X_train_vec, y_train, y_test, top_features):
    """Generate all 7 visualizations."""
    print("\n[*] Generating visualizations (7 plots)...")
    
    labels = sorted(data["genre"].unique())
    
    plot_genre_distribution(data)
    plot_word_clouds(data_processed)
    plot_model_comparison(results)
    plot_confusion_matrix(y_test, best_result["preds"], best_result["name"], labels)
    plot_learning_curve(best_result["model"], X_train_vec, y_train, best_result["name"])
    plot_top_features(top_features)
    plot_text_length_distribution(data)
    
    print(f"\n[OK] All 7 plots saved to: {PLOTS_DIR}")
