# Movie Genre Classification

A machine learning project that predicts movie genres from their plot descriptions. Built using Python, Scikit-learn, and NLTK.

---

## About

This project uses text classification techniques to categorize movies into genres based on their plot summaries. I used the CMU Movie Summary Corpus as the dataset and compared different ML models to find the best one.

**Genres supported:** Action, Comedy, Drama, Horror, Romance, Science Fiction, Thriller

---

## How It Works

1. Download real movie plot data from the CMU Movie Summary Corpus
2. Clean the text (lowercase, remove punctuation, lemmatize words)
3. Convert text to numbers using TF-IDF with bigrams
4. Train and compare 5 models using GridSearchCV
5. Save the best model so we don't have to retrain every time
6. Use the saved model to predict genres through a web interface

---

## Tech Stack

- **Python** - main language
- **Pandas** - data handling
- **Scikit-learn** - ML models, TF-IDF, GridSearchCV
- **NLTK** - text preprocessing (lemmatization, stop words)
- **Matplotlib & Seaborn** - charts and plots
- **WordCloud** - word cloud visualizations
- **Joblib** - saving/loading trained models
- **Streamlit** - web app frontend

---

## Project Structure

```
movie-genre-classification/
    data/
        fetch_dataset.py        # script to download dataset
        movies.csv              # processed dataset (2100 movies)
    src/
        preprocessing.py        # text cleaning and lemmatization
        feature_engineering.py  # TF-IDF vectorization
        models.py               # model training and saving
        visualization.py        # plot generation
        utils.py                # config and paths
    models/                     # saved trained models
    plots/                      # generated charts
    app.py                      # streamlit web app
    main.py                     # main training script
    requirements.txt
```

---

## Setup & Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download the dataset
```bash
python data/fetch_dataset.py
```
This downloads the CMU Movie Summary Corpus and processes it into 2100 balanced samples.

### Train the models
```bash
python main.py
```
This runs the full pipeline - preprocessing, feature extraction, model training with hyperparameter tuning, evaluation, and saves the best model.

### Launch the web app
```bash
streamlit run app.py
```
Opens a web interface where you can type any movie description and get a genre prediction.

---

## Dataset

- **Source:** CMU Movie Summary Corpus (Carnegie Mellon University)
- **Size:** 2,100 movie plot summaries
- **Balance:** 300 samples per genre
- **Preprocessing:** Lowercasing, punctuation removal, lemmatization, stop word removal

---

## Models Compared

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Linear SVM | 44.0% | 43.0% |
| Logistic Regression | 43.3% | 42.7% |
| Naive Bayes | 41.9% | 40.6% |
| Random Forest | 40.0% | 39.1% |
| Gradient Boosting | 33.6% | 33.4% |

Linear SVM performed the best with GridSearchCV tuning (C=0.1, squared_hinge loss).

---

## Visualizations

The training script generates 7 plots:
- Genre distribution chart
- Word clouds by genre
- Model accuracy comparison
- Confusion matrix
- Learning curve
- Top TF-IDF features per genre
- Text length distribution by genre

---

## What I Learned

- How to work with real-world text data and clean it for ML
- Difference between CountVectorizer and TF-IDF
- How bigrams improve text classification
- Using GridSearchCV to find the best hyperparameters
- Saving and loading models with joblib
- Building a web app with Streamlit

---

## Future Improvements

- Try deep learning models like LSTM or transformers
- Add multi-label classification (movies can have multiple genres)
- Use word embeddings (Word2Vec, GloVe) instead of TF-IDF
- Use a bigger dataset for better accuracy


## Live Demo / Deployment

A live deployment of this Streamlit app is available at:

https://movie-genre-classification-001.streamlit.app/

Feel free to try the web interface to input plot descriptions and get genre predictions.
