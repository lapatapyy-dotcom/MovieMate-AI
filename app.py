import streamlit as st
import pandas as pd
import requests
import os
from datetime import date, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# SETUP
# =========================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(
    page_title="Movie Recommendation AI",
    page_icon="🎬",
    layout="wide"
)


# =========================
# GET RECENT MOVIES FROM TMDB
# =========================

today = date.today()
six_months_ago = today - timedelta(days=180)

url = "https://api.themoviedb.org/3/discover/movie"

params = {
    "api_key": TMDB_API_KEY,
    "language": "en-US",
    "region": "IN",
    "sort_by": "popularity.desc",
    "primary_release_date.gte": six_months_ago.isoformat(),
    "primary_release_date.lte": today.isoformat(),
    "vote_count.gte": 5,
    "page": 1
}

response = requests.get(url, params=params)

if response.status_code != 200:
    st.error("Could not connect to TMDB. Please check your API key.")
    st.stop()

tmdb_data = response.json()
tmdb_movies = tmdb_data.get("results", [])

movies = pd.DataFrame(tmdb_movies)

if movies.empty:
    st.error("No movies were found.")
    st.stop()


# =========================
# GENRE NAMES
# =========================

genre_map = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    53: "Thriller",
    10752: "War",
    37: "Western"
}


movies["genre"] = movies["genre_ids"].apply(
    lambda ids: " ".join(
        genre_map.get(i, "")
        for i in ids
    )
)

movies["description"] = movies["overview"].fillna("")
movies["title"] = movies["title"].fillna("")
movies["poster_path"] = movies["poster_path"].fillna("")
movies["backdrop_path"] = movies["backdrop_path"].fillna("")
movies["release_date"] = movies["release_date"].fillna("")


# =========================
# CINEMATIC BACKGROUND
# =========================

background = ""

if movies.iloc[0]["backdrop_path"]:
    background = (
        "https://image.tmdb.org/t/p/original"
        + movies.iloc[0]["backdrop_path"]
    )

if background:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(5, 5, 15, 0.88),
                    rgba(5, 5, 15, 0.95)
                ),
                url("{background}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .movie-card {{
            background: rgba(20, 20, 35, 0.88);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.12);
        }}

        h1, h2, h3 {{
            color: white !important;
        }}

        p, label {{
            color: #dddddd !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================
# AI TEXT MODEL
# =========================

movies["movie_text"] = (
    movies["genre"] + " " +
    movies["description"]
)

vectorizer = TfidfVectorizer(stop_words="english")

movie_vectors = vectorizer.fit_transform(
    movies["movie_text"]
)


# =========================
# APP TITLE
# =========================

st.title("🎬 Movie Recommendation AI")

st.write(
    "Tell me your mood, preferred genre and what kind of story you want."
)

st.caption(
    f"🔥 Using {len(movies)} recent movies from TMDB"
)


# =========================
# USER INPUT
# =========================

st.subheader("😊 How are you feeling?")

mood = st.selectbox(
    "Choose your mood",
    [
        "Happy",
        "Sad",
        "Romantic",
        "Excited",
        "Relaxed",
        "Scared",
        "Emotional",
        "Motivational",
        "Curious",
        "Serious"
    ]
)


st.subheader("🎭 Choose a genre")

genre = st.selectbox(
    "Genre",
    [
        "Any",
        "Action",
        "Adventure",
        "Comedy",
        "Romance",
        "Drama",
        "Horror",
        "Thriller",
        "Science Fiction",
        "Fantasy",
        "Animation",
        "Mystery",
        "Crime",
        "Music"
    ]
)


st.subheader("📝 Describe the movie you want")

description = st.text_area(
    "What kind of movie are you looking for?",
    placeholder=(
        "Example: I want an exciting action movie "
        "with fighting, adventure and a surprising story."
    )
)


# =========================
# RECOMMENDATION
# =========================

if st.button("🎬 Recommend Movies"):

    if not description.strip():
        st.warning("Please describe what kind of movie you want.")

    else:

        user_text = f"{mood} {genre} {description}"

        user_vector = vectorizer.transform([user_text])

        similarity_scores = cosine_similarity(
            user_vector,
            movie_vectors
        ).flatten()

        results = movies.copy()

        results["score"] = similarity_scores

        # Genre filter
        if genre != "Any":
            results = results[
                results["genre"].str.contains(
                    genre,
                    case=False,
                    na=False
                )
            ]

        # Mood keyword matching
        mood_keywords = {
            "Happy": [
                "fun", "funny", "joy", "happy",
                "friend", "comedy", "adventure"
            ],
            "Sad": [
                "sad", "loss", "grief", "emotional",
                "life", "death"
            ],
            "Romantic": [
                "love", "romance", "relationship",
                "couple", "heart"
            ],
            "Excited": [
                "action", "fight", "battle",
                "adventure", "mission", "danger"
            ],
            "Relaxed": [
                "family", "friendship", "life",
                "feel-good"
            ],
            "Scared": [
                "horror", "terror", "killer",
                "survive", "danger", "mystery"
            ],
            "Emotional": [
                "love", "family", "life",
                "loss", "emotional", "relationship"
            ],
            "Motivational": [
                "dream", "success", "journey",
                "overcome", "struggle", "inspire"
            ],
            "Curious": [
                "mystery", "secret", "discover",
                "investigate", "unknown"
            ],
            "Serious": [
                "crime", "war", "politics",
                "drama", "investigation"
            ]
        }

        keywords = mood_keywords.get(mood, [])

        def mood_score(text):
            text = text.lower()
            return sum(
                1 for word in keywords
                if word in text
            )

        results["mood_score"] = results[
            "description"
        ].apply(mood_score)

        # Combine AI similarity + mood matching
        results["final_score"] = (
            results["score"] +
            results["mood_score"] * 0.08
        )

        results = results.sort_values(
            by="final_score",
            ascending=False
        )

        st.subheader("🍿 Recommended Movies")

        if results.empty:
            st.warning(
                "No movies matched that genre. "
                "Try selecting 'Any'."
            )

        else:

            for _, movie in results.head(5).iterrows():

                st.markdown(
                    '<div class="movie-card">',
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns([1, 3])

                with col1:

                    if movie["poster_path"]:
                        st.image(
                            "https://image.tmdb.org/t/p/w500"
                            + movie["poster_path"],
                            width=200
                        )

                with col2:

                    st.markdown(
                        f"### 🎬 {movie['title']}"
                    )

                    st.write(
                        f"**Genre:** {movie['genre']}"
                    )

                    st.write(
                        f"**Release Date:** "
                        f"{movie['release_date']}"
                    )

                    st.write(
                        f"**Description:** "
                        f"{movie['description']}"
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )