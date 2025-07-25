# 📰 Fake News Detection using NLP & Logistic Regression
Day 55 of LSPP
This project is a simple yet powerful fake news classifier built using:
- Natural Language Processing (TF-IDF)
- Logistic Regression
- Kaggle’s "Fake and Real News Dataset"

## 📂 Dataset
Downloaded from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
Used:
- `Fake.csv` → Fake news
- `True.csv` → Real news

## 🔍 How It Works
1. Loads and merges fake & real news CSVs
2. Cleans text data (removes punctuation, links, etc.)
3. Converts text to vectors using TF-IDF
4. Trains a Logistic Regression classifier
5. Evaluates performance using accuracy & classification report

## 🧪 Result
Achieved **~98% accuracy** on test data.

## 🛠 Tech Stack
- Python
- scikit-learn
- pandas
- TF-IDF Vectorizer

## 🚀 Run It Yourself
1. Clone the repo
2. Download dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
3. Place `Fake.csv` and `True.csv` in the same folder as `fake.py`
4. Run the script:

python fake.py
