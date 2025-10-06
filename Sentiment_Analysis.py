import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math

# -----------------------------
# --- 1. Custom VADER Lexicon ---
# -----------------------------
vader_lexicon = {
    'good': 1.9, 'great': 3.1, 'best': 3.2, 'excellent': 3.1, 'amazing': 3.0, 'nice': 1.8, 'love': 3.2,
    'bad': -2.5, 'worst': -3.2, 'poor': -2.1, 'terrible': -3.1, 'awful': -3.4, 'disappointing': -2.7,
    'average': 0.0, 'okay': 0.5, 'fine': 0.5, 'not': -0.7, 'no': -0.5, 'yes': 0.7, 'happy': 2.7,
    'sad': -2.3, 'satisfied': 2.4, 'unsatisfied': -2.2, 'recommend': 2.8, 'hate': -3.0
}

# -----------------------------
# --- 2. Custom Sentiment Analyzer ---
# -----------------------------
class CustomVADER:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def polarity_scores(self, text):
        if not isinstance(text, str):
            return {"compound": 0.0, "label": "Neutral"}
        words = re.findall(r'\w+', text.lower())
        score = sum(self.lexicon.get(w, 0) for w in words)
        norm = score / math.sqrt(len(words) + 1)
        if norm > 0.05:
            label = "Positive"
        elif norm < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return {"compound": round(norm, 4), "label": label}

analyzer = CustomVADER(vader_lexicon)

# -----------------------------
# --- 3. Helper Functions ---
# -----------------------------
def parse_reviews(review_str):
    """Convert stringified list of reviews to text."""
    if pd.isna(review_str):
        return ""
    try:
        review_list = ast.literal_eval(review_str)
        return " ".join(review_list)
    except:
        return str(review_str)

def build_query_search_system(df, query, top_k=3):
    """Compute TF-IDF similarity between query and product text."""
    df["Combined_Text"] = df["Description"].fillna("") + " " + df["User Reviews"].fillna("")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["Combined_Text"])
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df["Match Score"] = cosine_sim
    return df.sort_values(by="Match Score", ascending=False).head(top_k)

# -----------------------------
# --- 4. Load Dataset ---
# -----------------------------
df = pd.read_csv("product_dataset.csv")

# Clean review text
df["User Reviews"] = df["User Reviews"].apply(parse_reviews)

# Compute Sentiment for each product
df["Sentiment Score"] = df["User Reviews"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["Sentiment Summary"] = df["User Reviews"].apply(lambda x: analyzer.polarity_scores(x)["label"])

# -----------------------------
# --- 5. Category Keywords ---
# -----------------------------
category_keywords = {
    "mobile": "Mobiles",
    "phone": "Mobiles",
    "laptop": "Laptops",
    "headphone": "Headphones",
    "earphone": "Headphones",
    "watch": "Smartwatches",
    "smartwatch": "Smartwatches",
    "tablet": "Tablets"
}

# -----------------------------
# --- 6. Sentiment-Aware Search ---
# -----------------------------
def sentiment_aware_search(df, query, top_k=3):
    # Query sentiment
    qsent = analyzer.polarity_scores(query)
    qlabel = qsent["label"]
    qscore = qsent["compound"]
    print(f"Query Sentiment → {qlabel} (score={qscore})")

    # Detect category (fixed version)
    qlower = query.lower()
    words = re.findall(r'\w+', qlower)
    category = None
    for k, v in category_keywords.items():
        if k in words or k + "s" in words:  # handles plural like 'headphones'
            category = v
            break

    if category:
        print(f"Detected Category: {category}")
        df_filtered = df[df["Category"].str.lower() == category.lower()].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        print("No products found in the detected category.")
        return pd.DataFrame()

    # Get TF-IDF match scores
    search_results_df = build_query_search_system(df_filtered, query, top_k=len(df_filtered))

    # Merge sentiment data (keep match score)
    merged_df = pd.merge(
        search_results_df[["Product Name", "Category", "Price", "Rating", "Match Score"]],
        df_filtered[["Product Name", "Sentiment Summary", "Sentiment Score"]],
        on="Product Name",
        how="left"
    )

    # --- Sorting logic ---
    if qlabel == "Positive":
        merged_df = merged_df.sort_values(
            by=["Sentiment Score", "Match Score"], ascending=[False, False]
        )
    elif qlabel == "Negative":
        merged_df = merged_df.sort_values(
            by=["Sentiment Score", "Match Score"], ascending=[True, False]
        )
    else:  # Neutral
        merged_df["Sentiment Abs"] = merged_df["Sentiment Score"].abs()
        merged_df = merged_df.sort_values(
            by=["Sentiment Abs", "Match Score"], ascending=[True, False]
        )
        merged_df = merged_df.drop(columns=["Sentiment Abs"])

    return merged_df.head(top_k).reset_index(drop=True)

# -----------------------------
# --- 7. Example Usage ---
# -----------------------------
query = input("Enter your search query: ")
results = sentiment_aware_search(df, query, top_k=3)

print("=" * 100)
print(
    results[
        ["Product Name", "Price", "Rating", "Match Score", "Sentiment Summary", "Sentiment Score"]
    ].to_markdown(index=False, floatfmt=(".2f", ".2f", ".4f", "", ".4f"))
)
