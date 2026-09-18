import os

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "movies.csv")

# Ensure directories exist
for d in [DATA_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Genre list
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Science Fiction", "Thriller"]

# ML Config
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
N_GRAM_RANGE = (1, 2)  # unigrams + bigrams
CV_FOLDS = 5
