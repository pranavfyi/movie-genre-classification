# Streamlit web app for movie genre prediction
# Run: streamlit run app.py

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.utils import DATASET_PATH, PLOTS_DIR, MODELS_DIR
from src.preprocessing import TextPreprocessor
from src.models import load_model, predict_genre

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #b8b8d4;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
        margin: 1.5rem 0;
    }
    .prediction-card h2 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .prediction-card p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e8e8e8;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h3 {
        color: #333;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #888;
        font-size: 0.85rem;
        margin: 0.2rem 0 0;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }

    .sidebar .stMarkdown h3 {
        color: #333;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e8e8e8;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    """Load the trained model, vectorizer, and metadata."""
    model, vectorizer, metadata = load_model()
    if model is None:
        return None, None, None, None

    preprocessor = TextPreprocessor(
        remove_stopwords=metadata.get("preprocessor_config", {}).get("remove_stopwords", True),
        lemmatize=metadata.get("preprocessor_config", {}).get("lemmatize", True),
    )
    return model, vectorizer, preprocessor, metadata


@st.cache_data
def get_dataset():
    """Load the dataset for display."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


# ── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Movie Genre Classifier</h1>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, vectorizer, preprocessor, metadata = get_model()

    if model is None:
        st.error("**Model not found!** Please train the model first by running:")
        st.code("python data/fetch_dataset.py\npython main.py", language="bash")
        return

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        genres = metadata.get("genres", [])

        st.markdown("### Supported Genres")
        for genre in genres:
            st.markdown(f"- {genre}")

        st.markdown("---")
        st.markdown(
            "<p style='color:#aaa;font-size:0.8rem;text-align:center;'>"
            "CSA2001 - SEM 3 Project</p>",
            unsafe_allow_html=True
        )

        with st.expander("Technical Details", expanded=False):
            model_name = metadata.get("best_model_name", "Unknown")
            accuracy = metadata.get("accuracy", 0)
            f1 = metadata.get("f1_weighted", 0)
            params = metadata.get("best_params", {})

            st.metric("Model", model_name)
            st.metric("Accuracy", f"{accuracy*100:.1f}%")
            st.metric("F1 Score", f"{f1*100:.1f}%")
            st.markdown("**Parameters**")
            for k, v in params.items():
                st.text(f"  {k}: {v}")

    # ── Main Content ────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Predict Movie Genre")
        user_input = st.text_area(
            "Enter a movie plot description:",
            height=150,
            placeholder="e.g., A group of astronauts travel through a wormhole in search of a new habitable planet for humanity...",
        )

        predict_clicked = st.button("Predict Genre", use_container_width=True)

    with col2:
        st.markdown("### Quick Examples")
        examples = [
            ("A soldier fights in WWII behind enemy lines", "Action"),
            ("Friends go on a road trip with hilarious mishaps", "Comedy"),
            ("A detective solves a serial murder case", "Thriller"),
            ("Scientists discover alien life on Mars", "Sci-Fi"),
            ("A ghost haunts an old Victorian mansion", "Horror"),
        ]
        for desc, expected in examples:
            if st.button(f"Try: {desc[:45]}...", key=f"ex_{expected}"):
                user_input = desc
                predict_clicked = True

    # ── Prediction ──────────────────────────────────────────────────────
    if predict_clicked and user_input:
        with st.spinner("Analyzing..."):
            prediction, probabilities = predict_genre(user_input, model, vectorizer, preprocessor)

        # Prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <p>Predicted Genre</p>
            <h2>{prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Confidence scores
        if probabilities:
            st.markdown("#### Confidence Scores")

            sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])

            for genre, prob in sorted_probs:
                col_label, col_bar, col_pct = st.columns([2, 6, 1])
                with col_label:
                    st.markdown(f"**{genre}**")
                with col_bar:
                    st.progress(min(prob, 1.0))
                with col_pct:
                    st.markdown(f"{prob*100:.1f}%")

    elif predict_clicked:
        st.warning("Please enter a movie description.")


if __name__ == "__main__":
    main()

