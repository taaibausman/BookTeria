# 📄 app.py — BookTeria: Book Recommender
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import urllib.parse

# ========================
# 👑 App Configuration
# ========================
st.set_page_config(page_title="BookTeria", layout="wide")

CUSTOM_CSS = """
<style>
/* ===== GENERAL THEME ===== */
body {
    background-color: #fbe4ff;
    color: #4b0082;
    font-family: 'Comic Sans MS', cursive, sans-serif;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #f6d1f5 !important;
}

/* NAVIGATION ITEM HOVER EFFECT */
.st-emotion-cache-1y4p8pa a[data-testid="stSidebarNavLink"]:hover {
    background-color: #e6b3ff !important;
    color: black !important;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 0 8px #dda0dd;
    border-radius: 8px;
}

/* CURRENT NAVIGATION ITEM SELECTED: Replace red with purple + pulse glow */
.st-emotion-cache-1y4p8pa a[data-testid="stSidebarNavLinkActive"],
.st-emotion-cache-1y4p8pa a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: #c89aff !important;
    color: black !important;
    font-weight: bold;
    border-left: 6px solid #a557f3 !important;
    box-shadow:
        0 0 12px #b266ff,
        0 0 6px #dda0dd,
        0 0 20px rgba(186, 85, 211, 0.4);
    border-radius: 10px;
    animation: pulseGlow 2s infinite ease-in-out;
    transition: all 0.3s ease-in-out;
}

/* Glowing pulse animation */
@keyframes pulseGlow {
    0% {
        box-shadow: 0 0 10px rgba(178, 102, 255, 0.3),
                    0 0 20px rgba(186, 85, 211, 0.2);
    }
    50% {
        box-shadow: 0 0 20px rgba(178, 102, 255, 0.6),
                    0 0 30px rgba(186, 85, 211, 0.4);
    }
    100% {
        box-shadow: 0 0 10px rgba(178, 102, 255, 0.3),
                    0 0 20px rgba(186, 85, 211, 0.2);
    }
}

/* ===== BACKGROUND OF MAIN CONTENT ===== */
[data-testid="stAppViewContainer"] {
    background-color: #fbe4ff !important;
}

/* ===== HEADINGS ===== */
h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #8a2be2 !important;
}

/* ===== TEXT COLOR ADJUSTMENT (Outside black areas only) ===== */
.stMarkdown p,
[data-testid="stMarkdownContainer"] p,
.css-q8sbsg,
label {
    color: #666666 !important;
}

/* Input placeholders */
input::placeholder {
    color: #888888 !important;
}

/* Input text */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] * {
    color: #f5f0f0 !important;
}

/* ===== BUTTONS ===== */
button, .stButton>button {
    background-color: #dda0dd !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    box-shadow: 0 0 6px #c38cd4;
    transition: all 0.3s ease-in-out;
}

button:hover, .stButton>button:hover {
    background-color: #e8b8f1 !important;
    box-shadow: 0 0 12px #c38cd4, 0 0 6px #f9ccff;
}

/* ===== INPUTS ===== */
.stForm, .stSelectbox, .stTextInput, .stButton,
.css-1d391kg, .css-1v3fvcr, .css-hxt7ib {
    background-color: #f8d4f8 !important;
    border: 2px solid #caa5f2 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* ===== LINKS ===== */
[data-testid="stMarkdownContainer"] a,
a {
    color: #666666 !important;
    font-weight: bold;
    text-decoration: none;
}

a:hover {
    color: #ff69b4 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ========================
# 📦 Load Data and Models
# ========================
@st.cache_data
def load_model():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    return vectorizer, similarity_matrix

@st.cache_data
def load_books():
    return pd.read_csv("book_profiles.csv")

@st.cache_data
def load_metadata():
    return pd.read_csv("books.csv")

# ========================
# 📚 Genre Inference
# ========================
def infer_genre_from_profile(profile):
    profile = str(profile).lower()
    if any(word in profile for word in ["love", "romance", "relationship", "heart"]):
        return "Romance"
    elif any(word in profile for word in ["magic", "dragon", "fantasy", "wizard"]):
        return "Fantasy"
    elif any(word in profile for word in ["murder", "crime", "detective", "mystery"]):
        return "Mystery"
    elif any(word in profile for word in ["space", "alien", "future", "robot", "sci-fi"]):
        return "Science Fiction"
    elif any(word in profile for word in ["history", "war", "past", "ancient"]):
        return "Historical"
    elif any(word in profile for word in ["ghost", "horror", "haunted", "nightmare"]):
        return "Horror"
    elif any(word in profile for word in ["life", "journey", "inspirational", "memoir"]):
        return "Biography / Memoir"
    else:
        return "Unknown"

# ========================
# 🔄 Load and Merge
# ========================
vectorizer, similarity_matrix = load_model()
books = load_books()
metadata = load_metadata()

# Merge profiles with metadata
metadata = pd.merge(metadata, books[['book_id', 'profile']], on='book_id', how='left')

# Apply genre inference
metadata["genres"] = metadata["profile"].apply(infer_genre_from_profile)

# Create TF-IDF matrix for similarity comparisons
tfidf_matrix = vectorizer.transform(books['profile'])


# ========================
# 🧐 Helper Functions
# ========================

def get_book_details(title):
    # Capitalize and clean title and author for search query
    row = metadata[metadata['title'].str.lower() == title.lower()].iloc[0]
    clean_title = row['title'].strip().title()
    clean_author = row.get('authors', '').strip().title()

    # Build encoded search query
    search_query = f"{clean_title} {clean_author} site:amazon.com"
    encoded_query = urllib.parse.quote(search_query)
    search_url = "https://www.google.com/search?q=" + encoded_query

    # Add to row for consistent formatting
    row['buy_link'] = search_url
    row['title'] = clean_title
    row['authors'] = clean_author
    return row

def get_percent_liked(avg_rating):
    return round((avg_rating / 5.0) * 100, 1)

def show_book_card(title, author, image_url, percent_liked, link=None):
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(image_url, width=100)
    with col2:
        st.markdown(f"### {title}", unsafe_allow_html=True)
        st.markdown(f"_by {author}_")
        st.markdown(f"👍 Liked by **{percent_liked}%** of readers")
        if link and isinstance(link, str) and link.startswith("http"):
            st.markdown(f"<a href='{link}' target='_blank'><button style='background-color:#e6b3ff;color:black;padding:5px 10px;border:none;border-radius:8px;'>🛙️ Buy Now</button></a>", unsafe_allow_html=True)

def recommend_by_book(title, top_n=5):
    if title not in books['title'].values:
        return None
    idx = books[books['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in sim_scores[1:top_n+1]]
    return books.iloc[top_indices]

def recommend_by_interests(user_input, top_n=5):
    input_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][1:top_n+1]
    return books.iloc[top_indices]

# ========================
# 📚 Navigation Menu
# ========================
with st.sidebar:
    # Title in sidebar (already styled)
    st.markdown("<h1 style='text-align:center; color:#a052d4;'>🌸 BookTeria</h1>", unsafe_allow_html=True)

    # Option Menu with purple and centered "Navigation" title
    section = option_menu(
        menu_title="Navigation",
        options=["Select a Book", "Enter Interests", "Explore All Books", "Explore Data", "Buy Now", "About Us"],
        icons=["book", "lightbulb", "grid", "bar-chart", "cart", "person-circle"],
        menu_icon="stars",
        default_index=0,
        styles={
            "container": {
                "padding": "2px 5px 5px 5px",
                "background-color": "#f3d4fa"
            },
            "icon": {
                "color": "#a678d1",
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#f7c5e0",
                "color": "#7b2cbf"
            },
            "nav-link-selected": {
                "background-color": "#d8b4f8",
                "color": "#4a148c",
                "font-weight": "bold"
            },
            "menu-title": {  # 💜 Navigation title styling
                "color": "#7b2cbf",
                "font-weight": "bold",
                "font-size": "20px",
                "text-align": "center"
            }
        }
    )



st.markdown("<h1 style='text-align:center;color:#8a2be2;'>👑 Welcome to BookTeria 👑</h1>", unsafe_allow_html=True)

# ========================
# 📖 Book-Based Recommender
# ========================
if section == "Select a Book":
    st.title("📖 Recommend by Book")

    # Inject custom CSS and JavaScript to style Streamlit selectbox (inside iframe)
    st.markdown("""
        <style>
        /* Style select tag (for Firefox) */
        select {
            background-color: #fce3ff !important;
            color: black !important;
            font-family: 'Trebuchet MS', sans-serif;
            border: 2px solid #d8b4f8 !important;
            border-radius: 8px !important;
            padding: 8px;
        }

        /* Placeholder text color */
        option {
            color: #7b2cbf !important;
        }
        </style>

        <script>
        // Delay to wait for selectbox to render inside iframe
        setTimeout(() => {
            const iframe = document.querySelector('iframe');
            if (iframe) {
                const innerDoc = iframe.contentDocument || iframe.contentWindow.document;
                const selects = innerDoc.querySelectorAll('select');
                selects.forEach(select => {
                    select.style.backgroundColor = '#fce3ff';
                    select.style.color = 'black';
                    select.style.border = '2px solid #d8b4f8';
                    select.style.borderRadius = '8px';
                    select.style.fontFamily = 'Trebuchet MS';
                });
            }
        }, 1000);
        </script>
    """, unsafe_allow_html=True)

    title_list = books['title'].dropna().unique()

    with st.form("book_form"):
        selected_title = st.selectbox("Choose a book you like:", sorted(title_list), key="book_select")
        submitted = st.form_submit_button("🔍 Recommend Books")
        if submitted:
            results = recommend_by_book(selected_title)
            if results is not None and not results.empty:
                st.subheader(f"📘 Because you liked *{selected_title}*:")
                for _, row in results.iterrows():
                    details = get_book_details(row['title'])
                    show_book_card(
                        row['title'],
                        row['authors'],
                        details['image_url'],
                        get_percent_liked(details['average_rating']),
                        link=details.get("buy_link", "#")
                    )
            else:
                st.warning(f"🧘‍♀️ Oopsie-daisy! We couldn’t find *{selected_title}* in our royal collection. Maybe it’s in a different castle? 🏰✨ Try entering your interests instead to summon magical matches! 💫")
                if st.button("🔮 Switch to Interest-Based Search"):
                    section = "Enter Interests"
                    st.experimental_rerun()

# ========================
# 🧠 Interest-Based Recommender
# ========================
elif section == "Enter Interests":
    st.title("💡 Recommend by Interests")

    # Inject custom CSS
    # Inject custom CSS
    st.markdown("""
        <style>
        input[type="text"] {
            background-color: #f8ecff !important;
            border: 2px solid #c8a2c8 !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            color: black !important;  /* 👈 Changed from #4a148c to black */
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 16px !important;
            box-shadow: 2px 2px 5px rgba(170, 120, 200, 0.2);
        }
    
        input::placeholder {
            color: #b288d1 !important;
            opacity: 0.8;
        }
        </style>
    """, unsafe_allow_html=True)


    with st.form("interest_form"):
        user_input = st.text_input("What do you love? (e.g., magic, dragons, love)")
        submitted = st.form_submit_button("🔍 Find Matches")

        if submitted and user_input:
            results = recommend_by_interests(user_input)
            st.subheader("📘 Books based on your interests:")
            if not results.empty:
                for _, row in results.iterrows():
                    details = get_book_details(row['title'])
                    show_book_card(row['title'], row['authors'], details['image_url'], get_percent_liked(details['average_rating']), link=details.get("buy_link", "#"))
            else:
                st.warning("🌟 No exact match, but here’s something close! Want us to look online in our magical realm? 🧹")

# ========================
# 📚 Explore All Books
# ========================
elif section == "Explore All Books":
    st.title("📚 Browse All Books in BookTeria")
    for _, row in metadata.iterrows():
        genre = row.get('genres', 'Unknown')
        details = get_book_details(row['title'])
        show_book_card(row['title'], row['authors'], row['image_url'], get_percent_liked(row['average_rating']), link=details.get("buy_link", "#"))
        st.markdown(f"📖 Genre/Type: *{genre}*")
        st.markdown("---")

# ========================
# 📊 EDA Visuals
# ========================
elif section == "Explore Data":
    # Inject CSS to make all text black
    st.markdown("""
        <style>
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Explore the World of Books")
    st.image("outputs/avg_rating_dist.png", caption="🌟 How Readers Rated Books", use_container_width=True)
    st.image("outputs/publication_years.png", caption="📅 Publication Timeline Over Years", use_container_width=True)
    st.image("outputs/language_dist.png", caption="🌍 Preferred Languages of Readers", use_container_width=True)
    st.image("outputs/rating_vs_reviews.png", caption="🔣️ Relation Between Ratings and Reviews", use_container_width=True)
    st.image("outputs/top_tags.png", caption="🍿 Most Popular Tags by Readers", use_container_width=True)
    st.image("outputs/tags_per_book_dist.png", caption="📚 How Many Tags Each Book Gets", use_container_width=True)

# ========================
# 📦 Buy Now
# ========================
elif section == "Buy Now":
    st.title("📦 Buy Your Favorite Book with Single Word Search")

    st.markdown("""
        <style>
        input[type="text"] {
            background-color: #f8ecff !important;
            border: 2px solid #c8a2c8 !important;
            border-radius: 10px !important;
            padding: 8px 12px !important;
            color: black !important;
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 16px !important;
            box-shadow: 2px 2px 5px rgba(170, 120, 200, 0.2);
        }

        input::placeholder {
            color: #b288d1 !important;
            opacity: 0.8;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("buy_now_form"):
        search_title = st.text_input("Enter one word from the book title:")

        submitted = st.form_submit_button("🔍 Search for Purchase")
        if submitted and search_title:
            match = metadata[metadata['title'].str.lower().str.contains(search_title.strip().lower(), na=False)]
            if not match.empty:
                row = match.iloc[0]
                buy_link = get_book_details(row['title'])['buy_link']
                genre = row.get("genres", "Unknown")
                st.success(f"🎉 Found it! Here's your royal link to buy *{row['title']}*: 👑")
                st.markdown(f"📖 Genre/Type: *{genre}*")
                st.markdown(f"""
                    <a href='{buy_link}' target='_blank'>
                        <button style='background-color:#dda0dd;color:white;padding:10px 15px;border:none;border-radius:8px;'>
                            💻 Buy Now on Amazon
                        </button>
                    </a>
                """, unsafe_allow_html=True)
            else:
                st.warning("👑 Alas! This book is not yet in our royal library. Please try another title!")
                suggestions = metadata[metadata['title'].str.lower().str.contains(search_title.strip().lower().split()[0], na=False)].head(3)
                if not suggestions.empty:
                    st.markdown("🔍 Perhaps you meant:")
                    for _, row in suggestions.iterrows():
                        genre = row.get("genres", "Unknown")
                        st.markdown(
                            f"""<p style='color:#555;'>- 📘 <em>{row['title']}</em> by {row['authors']} | <strong>Genre:</strong> {genre}</p>""",
                            unsafe_allow_html=True
                        )


# ========================
# 👑 About Us
# ========================
elif section == "About Us":
    st.title("👑 About BookTeria")
    st.markdown("""
        Welcome to **BookTeria** — your Book explorer. Created by **Taaiba Usman**, this dreamy app helps you discover magical reads based on what you love.
        Whether you adore mystery, fantasy, romance or adventure — BookTeria has a magical match for you 💜  
        **Dream Big. Read Often. Rule your Kingdom of Imagination.** 📖👸
    """)
