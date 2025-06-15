---
title: 📚 BookTeria — GoodBooks Explorer
emoji: 📖
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
---

# 📘 BookTeria — Content-Based Book Recommender

BookTeria is a content-based book recommendation system powered by the **GoodBooks-10k dataset** (10K books, 1M+ ratings).

It lets users discover books in two interactive ways:

- 🔖 Select a book they already liked
- 🧠 Type in their reading interests (e.g., “romance, magic, dragons”)

---

## 🔧 How It Works

- **TF-IDF Vectorizer** is trained on book metadata (title + author + tags)
- **Cosine Similarity** finds the most similar books
- **Streamlit app** offers a responsive, interactive user experience

---

## 🧠 Features

- 📘 Dual recommendation modes (by book or interest)
- 🛍️ “Buy Now” button to explore books externally
- 📊 EDA Insights — Ratings, tags, languages, and reviews
- 📚 Explore All Books with popularity and genre
- 💬 Friendly fallbacks if a book isn't found
- ✅ Runs entirely on local files — no internet needed after setup

---

## 📂 Included Files

- `app.py` — Streamlit app
- `book_profiles.csv` — Preprocessed book profiles
- `vectorizer.pkl` — TF-IDF model
- `model.pkl` — Cosine similarity matrix
- `books.csv`, `tags.csv`, `book_tags.csv` — Raw dataset files
- `outputs/` — EDA graphs used in the app

---

## 📊 Dataset Source

- [GoodBooks-10k on Kaggle](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
