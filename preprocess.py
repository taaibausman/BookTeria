# preprocess.py — Build Book Profiles
# ===========================
# This script merges books, authors, titles, and tags into a single 'profile'
# per book. These profiles are later vectorized in model.py to build a
# content-based recommendation system.

import pandas as pd

# Step 1: Load all relevant CSV files
# Assumes all files are in the same directory as the script
# - books.csv: contains book metadata
# - tags.csv: maps tag_id to tag_name
# - book_tags.csv: lists how often a tag is used for each book
def load_data():
    books = pd.read_csv("books.csv")
    tags = pd.read_csv("tags.csv")
    book_tags = pd.read_csv("book_tags.csv")
    return books, tags, book_tags

# Step 2: Get top N most common tags for each book
# - Filters to top N frequent tags globally (e.g., top 1000)
# - Groups tags per book into a single string (e.g., "fantasy magic young-adult")
# Returns a DataFrame: { goodreads_book_id, tag_string }
def get_book_tags(tags, book_tags, top_n_tags=1000):
    # Merge to get tag_name instead of tag_id
    merged = book_tags.merge(tags, on='tag_id')

    # Keep only top N most used tags globally
    top_tags = merged['tag_name'].value_counts().head(top_n_tags).index
    filtered = merged[merged['tag_name'].isin(top_tags)]

    # Group tag names by book
    grouped = filtered.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
    grouped.columns = ['goodreads_book_id', 'tag_string']

    return grouped

# Step 3: Merge book metadata with tags
# - Builds a unified profile per book: title + author + tag_string
# - Saves the result to 'book_profiles.csv'
def build_book_profiles():
    books, tags, book_tags = load_data()
    tag_data = get_book_tags(tags, book_tags)

    # Merge tag strings into books table using best_book_id
    books_profiles = books.merge(tag_data, how='left', left_on='best_book_id', right_on='goodreads_book_id')

    # Handle books without tags by filling empty tag_string
    books_profiles['tag_string'] = books_profiles['tag_string'].fillna('')

    # Create 'profile' = title + author + tags
    books_profiles['profile'] = (
        books_profiles['title'].fillna('') + ' ' +
        books_profiles['authors'].fillna('') + ' ' +
        books_profiles['tag_string']
    )

    # Keep only relevant columns
    final_df = books_profiles[['book_id', 'title', 'authors', 'profile']]

    # Save to CSV for use in model training
    final_df.to_csv("book_profiles.csv", index=False)
    print("✅ Book profiles built and saved to 'book_profiles.csv'")

    return final_df

# Entry point for script execution
if __name__ == "__main__":
    build_book_profiles()
