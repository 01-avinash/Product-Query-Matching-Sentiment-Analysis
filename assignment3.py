%%writefile app.py
# ===============================================
# 📱 Streamlit AI-Powered Mobile Search & Sentiment System
# ===============================================

import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -----------------------------------------------
# 🏷️ Page Setup
# -----------------------------------------------
st.set_page_config(page_title="AI Mobile Finder", page_icon="📱", layout="wide")
st.title("📱 AI-Powered Mobile Search & Sentiment System")
st.markdown("Search mobiles using natural language queries — powered by **AI embeddings + sentiment analysis**.")

# -----------------------------------------------
# 📂 Load Dataset
# -----------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("top_mobiles 25.csv")  # ✅ Make sure file name is correct

df = load_data()
st.success(f"✅ Dataset loaded with {len(df)} records")

# -----------------------------------------------
# 🤖 Load AI Models (Embeddings + Sentiment)
# -----------------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return embedder, sentiment_model

st.info("🔹 Loading AI models... please wait")
embedder, sentiment_model = load_models()

# -----------------------------------------------
# 🧠 Preprocess Data
# -----------------------------------------------
@st.cache_data(show_spinner=True)
def preprocess_data(df):
    df["Combined_Text"] = (
        df["Description"].fillna('') + " " +
        df["User_Review_Excerpt"].fillna('') + " " +
        df["Features"].fillna('')
    )

    product_embeddings = embedder.encode(df["Combined_Text"].tolist(), convert_to_tensor=True)

    # Sentiment analysis
    df["Sentiment"] = df["User_Review_Excerpt"].fillna('').apply(
        lambda x: sentiment_model(x[:512])[0]['label'] if len(x) > 0 else "neutral"
    )
    df["Sentiment_Score"] = df["User_Review_Excerpt"].fillna('').apply(
        lambda x: sentiment_model(x[:512])[0]['score'] if len(x) > 0 else 0.0
    )

    return df, product_embeddings

df, product_embeddings = preprocess_data(df)
st.success("✅ Data encoded and sentiment analyzed")

# -----------------------------------------------
# 🔍 AI-Powered Mobile Search
# -----------------------------------------------
def ai_mobile_search(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    df["Similarity_Score"] = cosine_scores.cpu().numpy()

    # Hybrid Score Calculation
    df["Hybrid_Score"] = (
        (df["Similarity_Score"] * 0.5) +
        (df["Sentiment_Score"] * 0.2) +
        (df["Rating_out_of_5"] / 5 * 0.3)
    )

    return df.sort_values(by="Hybrid_Score", ascending=False).head(top_k).reset_index(drop=True)

# -----------------------------------------------
# 🧩 Smart Multi-Keyword Column Selection
# -----------------------------------------------
def select_columns_by_query(query):
    query_lower = query.lower()

    # Extended mapping with synonyms
    keyword_mapping = {
        "Features": ["feature", "spec", "specification", "details"],
        "Camera Quality": ["camera", "photo", "picture", "selfie"],
        "Battery Quality": ["battery", "charge", "charging", "performance", "power"],
        "Price_INR": ["price", "cost", "budget", "cheap", "affordable", "under", "less than"],
        "Rating_out_of_5": ["rating", "review score", "stars", "good rating", "feedback"],
        "Description": ["description", "overview", "info", "information"],
        "User_Review_Excerpt": ["review", "user feedback", "opinion"],
        "Sentiment": ["sentiment", "positive", "negative", "neutral", "emotion"]
    }

    selected_columns = []

    # Match any keyword in the query
    for col, keywords in keyword_mapping.items():
        for word in keywords:
            if word in query_lower:
                selected_columns.append(col)
                break  # avoid duplicates if multiple words from same category match

    # Default columns if nothing matched
    if not selected_columns:
        selected_columns = ["Features", "Price_INR", "Rating_out_of_5"]

    return selected_columns

# -----------------------------------------------
# 🧭 User Interface
# -----------------------------------------------
query = st.text_input("🔍 Enter your search query:", "best camera phone under 50000")

if st.button("Search"):
    with st.spinner("🤖 Searching AI-powered matches..."):
        results = ai_mobile_search(query, top_k=5)
        selected_columns = select_columns_by_query(query)

        if not results.empty:
            st.subheader(f"📊 Top Mobiles — Showing: **{', '.join(selected_columns)}**")

            # Always include product name + hybrid score
            display_cols = ["Product Name"] + selected_columns + ["Hybrid_Score"]

            st.dataframe(results[display_cols])

            results[display_cols].to_csv("ai_mobile_search_results_filtered.csv", index=False)
            st.success("💾 Results saved as `ai_mobile_search_results_filtered.csv`")
        else:
            st.warning("⚠️ No matching results found. Try a different query.")
