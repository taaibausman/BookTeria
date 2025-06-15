# eda.py — Exploratory Data Analysis for GoodBooks-10k
# ===========================
# This script performs EDA on the books and tag-related data to uncover patterns,
# trends, and distributions that help in building a content-based recommendation system.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create output directory if not exists to store all plots
os.makedirs("outputs", exist_ok=True)

# Load all necessary datasets
# - books.csv: contains metadata about books
# - tags.csv: maps tag_id to tag_name
# - book_tags.csv: contains tag counts per book (many-to-many relationship)
def load_data():
    books = pd.read_csv("books.csv")
    tags = pd.read_csv("tags.csv")
    book_tags = pd.read_csv("book_tags.csv")
    return books, tags, book_tags

# Perform EDA on books.csv
# This explores ratings, languages, authors, and other metadata
def books_eda(books):
    print("\n=== Basic Info ===")
    print(books.info())
    print("\n=== Summary Stats ===")
    print(books.describe(include='all'))

    # 1. Top 10 most rated books
    top_rated = books.sort_values(by='ratings_count', ascending=False).head(10)
    print("\nTop 10 Most Rated Books:\n", top_rated[['title', 'authors', 'ratings_count']])

    # 2. Histogram of average ratings
    plt.figure(figsize=(8,5))
    sns.histplot(books['average_rating'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Average Ratings')
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Books')
    plt.tight_layout()
    plt.savefig("outputs/avg_rating_dist.png")
    plt.close()

    # 3. Top 10 authors by number of books
    top_authors = books['authors'].value_counts().head(10)
    print("\nTop 10 Authors by Book Count:\n", top_authors)

    # 4. Histogram of publication years (shows how books are distributed across years)
    plt.figure(figsize=(10,5))
    sns.histplot(books['original_publication_year'].dropna(), bins=50, kde=False, color='salmon')
    plt.title('Books Published Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Books')
    plt.tight_layout()
    plt.savefig("outputs/publication_years.png")
    plt.close()

    # 5. Top 10 languages
    lang_counts = books['language_code'].value_counts().head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(
        x=lang_counts.index,
        y=lang_counts.values,
        hue=lang_counts.index,
        dodge=False,
        palette='Set2',
        legend=False
    )
    plt.title('Top 10 Languages Used')
    plt.xlabel('Language Code')
    plt.ylabel('Number of Books')
    plt.tight_layout()
    plt.savefig("outputs/language_dist.png")
    plt.close()

    # 6. Scatter plot of average rating vs text review count
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='average_rating', y='work_text_reviews_count', data=books, alpha=0.5)
    plt.title('Rating vs Text Review Count')
    plt.xlabel('Average Rating')
    plt.ylabel('Text Review Count')
    plt.tight_layout()
    plt.savefig("outputs/rating_vs_reviews.png")
    plt.close()

# Perform EDA on tag data
# Helps us understand what tags are popular and how many tags are used per book

def tags_eda(tags, book_tags):
    # Merge tag_id with tag_name for clarity
    merged_tags = book_tags.merge(tags, on='tag_id')

    # 1. Count top 20 most frequently used tags across all books
    tag_counts = merged_tags['tag_name'].value_counts().head(20)
    print("\nTop 20 Most Frequent Tags:\n", tag_counts)

    # Bar plot for top 20 tags
    plt.figure(figsize=(10,6))
    sns.barplot(
        x=tag_counts.values,
        y=tag_counts.index,
        hue=tag_counts.index,
        dodge=False,
        palette="viridis",
        legend=False  # avoids unnecessary duplicate legend
    )
    plt.title("Top 20 Most Used Tags")
    plt.xlabel("Tag Count")
    plt.ylabel("Tag Name")
    plt.tight_layout()
    plt.savefig("outputs/top_tags.png")
    plt.close()


    # 2. Histogram of number of tags per book (how richly tagged each book is)
    tag_per_book = merged_tags.groupby('goodreads_book_id')['tag_name'].count()
    plt.figure(figsize=(8,5))
    sns.histplot(tag_per_book, bins=30, color='orange')
    plt.title("Distribution of Number of Tags per Book")
    plt.xlabel("Number of Tags")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig("outputs/tags_per_book_dist.png")
    plt.close()

# Main driver
# Load the data, and run both EDA parts
if __name__ == "__main__":
    books, tags, book_tags = load_data()
    books_eda(books)
    tags_eda(tags, book_tags)
