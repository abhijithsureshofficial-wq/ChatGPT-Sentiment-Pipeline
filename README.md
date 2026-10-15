# ChatGPT App Store Sentiment Analysis Pipeline

An end-to-end, production-grade NLP pipeline for multi-class sentiment analysis on ChatGPT Google Play Store reviews.

## Overview
App store reviews present a unique set of NLP challenges: they are highly informal, heavily reliant on emojis, and suffer from severe class imbalance (the "U-shaped" distribution of 1-star and 5-star reviews).
This project is a production-grade, end-to-end sentiment analysis pipeline designed specifically to tackle these real-world data issues. Rather than relying on out-of-the-box models, this pipeline engineers features to handle edge cases: it translates emojis into class-aligned text tokens, preserves critical negations (e.g., "not good"), and utilizes cost-sensitive modeling to ensure the minority neutral class isn't ignored.
Finally, to guarantee production reliability, the entire workflow—from raw text cleaning to TF-IDF vectorization and classification—is encapsulated in a single Scikit-Learn Pipeline artifact, ensuring absolute train-serve parity.

## Features
- **Advanced Preprocessing:** Custom text cleaner that translates emojis into sentiment tokens, preserves negations, and expands contractions.
- **Cost-Sensitive Modeling:** Mitigates severe class imbalance (common in app reviews) using Macro-F1 optimization and balanced Logistic Regression.
- **End-to-End Pipeline:** Wraps text cleaning, TF-IDF vectorization, and classification into a single Scikit-Learn `Pipeline` to eliminate train-serve skew.
- **Error Diagnostics:** Includes false-positive/negative analysis to surface model blind spots.

## Tech Stack
- Python, Scikit-Learn, NLTK, Pandas
- TF-IDF, Multinomial Naive Bayes, Logistic Regression

## Project Structure
- `notebooks/`: Contains the full EDA, training, and evaluation pipeline.
- `requirements.txt`: Dependencies required to run the project.

## Project Architecture
Raw Review Text 
   │
   ▼
[ ReviewCleaner ] ──► (Lowercases, translates emojis, expands contractions, keeps negations)
   │
   ▼
[ TF-IDF Vectorizer ] ──► (Unigrams + Bigrams, sublinear scaling, noise filtering)
   │
   ▼
[ Balanced Classifier ] ──► (Logistic Regression / Naive Bayes)
   │
   ▼
Predicted Sentiment + Confidence Score

## How to Run
1. Clone the repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/chatgpt_sentiment.ipynb` in Jupyter or VS Code.
4. Upload the `chatgpt_reviews.csv` dataset when prompted in Cell 2.
