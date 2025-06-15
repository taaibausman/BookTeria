# model.py — Train Content-Based Recommendation Model
# ===========================
# This script loads book_profiles.csv, vectorizes the content using TF-IDF,
# computes cosine similarity, and saves both the vectorizer and similarity matrix.

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the book profiles generated from preprocess.py
def load_profiles():
    return pd.read_csv("book_profiles.csv")

# Build and train TF-IDF(Term Frequency – Inverse Document Frequency) 
# Similarity matrix(Cosine Similarity -> Measures the angle between two TF-IDF vectors. The smaller the angle, the more similar the content.)
def train_model():
    df = load_profiles()

    # Step 1: Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Step 2: Vectorize the 'profile' text
    tfidf_matrix = vectorizer.fit_transform(df['profile'])

    # Step 3: Compute cosine similarity between all books
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 4: Save the model and vectorizer using pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(similarity_matrix, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
    return df, similarity_matrix

# Function to recommend similar books given a title
def recommend_books(book_title, top_n=5):
    df = load_profiles()
    with open("model.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)

    # Get the index of the given book
    if book_title not in df['title'].values:
        print(f"❌ '{book_title}' not found in book list.")
        return []

    idx = df[df['title'] == book_title].index[0]

    # Get similarity scores and sort them
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar books excluding itself
    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    recommendations = df.iloc[top_indices][['title', 'authors']]
    return recommendations

# Run model training if script is executed directly
if __name__ == "__main__":
    train_model()
