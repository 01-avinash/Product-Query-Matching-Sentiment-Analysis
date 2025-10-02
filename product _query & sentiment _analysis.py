import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math

# --- VADER Lexicon and Analyzer (Custom Implementation) ---
vader_lexicon = {
    'good': 1.9, 'great': 3.1, 'best': 3.2, 'excellent': 3.1, 'amazing': 2.7, 'well': 1.6,
    'love': 2.4, 'highly': 1.6, 'recommend': 1.4, 'fast': 1.2, 'awesome': 3.2,
    'poor': -2.2, 'bad': -2.5, 'not': -0.4, 'disappointing': -2.3, 'drains': -1.2,
    'fail': -2.4, 'worst': -3.4, 'cheap': -1.5, 'average': 0.0, 'ok': 0.8,
    'strong': 1.8, 'clear': 1.4, 'camera': 0.0, 'phone': 0.0, 'battery': 0.0, 'money': 0.0
}

class CustomVaderAnalyzer:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def polarity_scores(self, text):
        words = re.findall(r'\w+', text.lower())
        vs = [self.lexicon.get(word, 0) for word in words]

        if not vs:
            return {'compound': 0.0, 'label': 'Neutral'}

        compound = sum(vs)
        compound = compound / math.sqrt(len(vs) * 2)

        if compound >= 0.05:
            sentiment_label = 'Positive'
        elif compound <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        return {'compound': round(compound, 4), 'label': sentiment_label}

analyzer = CustomVaderAnalyzer(vader_lexicon)

# --- 1. Data Loading and Preparation ---
df = pd.read_csv("product_dataset.csv")

def parse_reviews(review_str):
    if pd.isna(review_str):
        return ""
    try:
        review_list = ast.literal_eval(review_str)
        return " ".join(review_list)
    except:
        return str(review_str)

df["Cleaned_Reviews"] = df["User Reviews"].apply(parse_reviews)
df["Search_Text"] = df["Description"].fillna("") + " " + df["Cleaned_Reviews"]

# --- 2. Query-Based Search System ---
def build_query_search_system(df: pd.DataFrame, query: str, top_k: int = 3) -> pd.DataFrame:
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["Search_Text"])

    query_vec = tfidf_vectorizer.transform([query])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results_df = df.iloc[top_k_indices].copy()
    results_df["Match Score"] = cosine_scores[top_k_indices]

    return results_df

# --- 3. Sentiment Analysis on Products ---
def calculate_product_sentiment(reviews_text, analyzer):
    if pd.isna(reviews_text) or not str(reviews_text).strip():
        return 0.0, "Neutral"

    score = analyzer.polarity_scores(str(reviews_text))
    return score["compound"], score["label"]

df[["Sentiment Score", "Sentiment Summary"]] = df["Cleaned_Reviews"].apply(
    lambda x: pd.Series(calculate_product_sentiment(x, analyzer))
)

# --- 4. Sentiment-Aware Search ---
def sentiment_aware_search(df, query, top_k=3):
    # Step 1: Get query sentiment
    query_sentiment = analyzer.polarity_scores(query)
    query_label = query_sentiment["label"]
    print(f"Query Sentiment → {query_label} (score={query_sentiment['compound']})")

    # Step 2: Run normal semantic search
    search_results_df = build_query_search_system(df, query, top_k=len(df))  # all matches

    # Step 3: Merge with sentiment
    merged_df = pd.merge(
        search_results_df[["Product Name", "Price", "Rating", "Match Score"]],
        df[["Product Name", "Sentiment Summary", "Sentiment Score"]],
        on="Product Name",
        how="left"
    )

    # Step 4: Filter based on query sentiment
    if query_label == "Positive":
        filtered = merged_df[merged_df["Sentiment Summary"] == "Positive"]
    elif query_label == "Negative":
        filtered = merged_df[merged_df["Sentiment Summary"] == "Negative"]
    else:  # Neutral query
        filtered = merged_df[merged_df["Sentiment Summary"] == "Neutral"]

    # Step 5: Fallback → if not enough results, show top_k regardless
    if len(filtered) < top_k:
        filtered = merged_df.head(top_k)

    return filtered.head(top_k)

# --- 5. Run the Search ---
search_query = input("Enter your search query: ")
final_results = sentiment_aware_search(df, search_query, top_k=3)

print("=" * 80)
print(f"Final Results for Query: '{search_query}'")
print("=" * 80)
print(
    final_results[
        ["Product Name", "Price", "Rating", "Match Score", "Sentiment Summary", "Sentiment Score"]
    ].to_markdown(index=False, floatfmt=(".2f", ".2f", ".2f", ".4f", "", ".4f"))
)