# 📉 Corporate Reputation Analysis using Sentiment Analysis

## 📝 Description

This project evaluates the impact of corporate layoffs on brand reputation using sentiment analysis on tweets before, during, and after layoffs. It employs NLP preprocessing, a trained machine learning model, and Net Brand Reputation (NBR) scoring to visualize sentiment trends.

## 🛠️ Tech Stack

- Python
- Google Colab
- Pandas, NumPy
- Scikit-learn
- NLTK, TextBlob, Contractions
- TfidfVectorizer
- Matplotlib & Seaborn
- Pretrained Sentiment Classifier (Pickle model)

## ✅ Features

- Uploads and processes tweet datasets (`before layoff`, `during layoff`, `after layoff`)
- Preprocesses text (cleaning, stopword removal, lemmatization)
- Applies pretrained ML model to classify sentiment (Positive, Negative, Neutral)
- Computes Net Brand Reputation (NBR) score
- Visualizes results using bar charts, pie charts, and count plots



## 🚀 How to Run

1. Open the notebook or script in **Google Colab**.
2. Upload the following files when prompted:
   - `tweets_before.csv` (or similar)
   - `layoffs.csv`
   - `tweets_after.csv`
   - Pretrained sentiment model (e.g., `sentiment_model.pkl`)
   - TF-IDF vector training dataset
3. All visualizations and reputation metrics will be displayed as output cells.

## 📈 Net Brand Reputation Formula

\[
\text{NBR} = \left( \frac{\text{Positive} - \text{Negative}}{\text{Positive} + \text{Negative}} \right) \times 100
\]

## 📂 Dataset Structure

Each dataset must include:
- `Tweet` column (cleaned automatically)
- Optionally: `User`, `Date Created` (dropped during cleaning)

---

*Author: Tanuja Subhash Shinde*
