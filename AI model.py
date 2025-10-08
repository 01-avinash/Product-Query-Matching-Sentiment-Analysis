
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rapidfuzz import fuzz

# -----------------------------------------------
#  Load Dataset
# -----------------------------------------------
df = pd.read_csv("top_mobiles 25.csv")
print(f" Dataset loaded with {len(df)} records")

# -----------------------------------------------
#  Load Pretrained Models
# -----------------------------------------------
print(" Loading models... please wait")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
print(" Models loaded successfully")

# -----------------------------------------------
#  Preprocess Data
# -----------------------------------------------
def preprocess_data(df):
    df["Combined_Text"] = (
        df["Description"].fillna("") + " " +
        df["User_Review_Excerpt"].fillna("") + " " +
        df["Features"].fillna("")
    )

    product_embeddings = embedder.encode(df["Combined_Text"].tolist(), convert_to_tensor=True)

    # Sentiment analysis
    sentiments, scores = [], []
    for text in df["User_Review_Excerpt"].fillna("").tolist():
        if len(text.strip()) == 0:
            sentiments.append("neutral")
            scores.append(0.0)
        else:
            result = sentiment_model(text[:512])[0]
            sentiments.append(result["label"].capitalize())
            scores.append(result["score"])

    df["Sentiment"] = sentiments
    df["Sentiment_Score"] = scores
    return df, product_embeddings


df, product_embeddings = preprocess_data(df)
print(" Data preprocessed and embeddings generated")

# -----------------------------------------------
#  Feature Keywords + Matching Logic
# -----------------------------------------------
keyword_mapping = {
    "Camera Quality": ["camera", "photo", "picture", "cam", "image", "selfie", "camra", "camora", "comora", "kamra"],
    "Battery Quality": ["battery", "charging", "charger", "power", "backup", "battry", "battary", "batri"],
    "Display": ["screen", "display", "resolution", "brightness", "amoled", "visual effect", "dispaly", "diplay"],
    "Performance": ["speed", "processor", "chip", "ram", "storage", "performance", "fast", "gaming"],
    "Features": ["phone", "mobile", "features", "design", "build", "look", "android", "cell", "iphone", "apple"],
    "Price_INR": ["price", "budget", "cost", "amount", "prize"],
}

def fuzzy_match(word, keywords, threshold=75):
    best_match = None
    best_score = 0
    for k in keywords:
        score = fuzz.ratio(word, k)
        if score > best_score:
            best_score = score
            best_match = k
    if best_score >= threshold:
        return True, best_match, best_score
    return False, None, best_score


def semantic_keyword_match(query, keyword_list, threshold=0.6):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    keyword_embs = embedder.encode(keyword_list, convert_to_tensor=True)
    sims = util.cos_sim(query_emb, keyword_embs)
    max_sim = torch.max(sims).item()
    if max_sim >= threshold:
        match_index = torch.argmax(sims).item()
        return True, keyword_list[match_index], max_sim
    return False, None, max_sim


# -----------------------------------------------
#  Detect Feature Intent (Fuzzy + Semantic)
# -----------------------------------------------
def detect_feature_intent(query, keyword_mapping):
    selected_columns = []
    match_reasons = []
    found_matches = []

    for col, keywords in keyword_mapping.items():
        for w in query.lower().split():
            fuzzy_ok, fuzzy_kw, fuzzy_score = fuzzy_match(w, keywords)
            semantic_ok, semantic_kw, semantic_score = semantic_keyword_match(w, keywords)

            if fuzzy_ok:
                selected_columns.append(col)
                found_matches.append((w, fuzzy_kw, col, round(fuzzy_score, 2)))
                match_reasons.append(f"Fuzzy matched '{w}' ≈ '{fuzzy_kw}' ({col}) [Score={round(fuzzy_score,2)}]")
                break
            elif semantic_ok:
                selected_columns.append(col)
                found_matches.append((w, semantic_kw, col, round(semantic_score, 2)))
                match_reasons.append(f"Semantic matched '{w}' ≈ '{semantic_kw}' ({col}) [Score={round(semantic_score,2)}]")
                break

    if not selected_columns:
        selected_columns = ["Features", "Price_INR", "Rating_out_of_5"]
        match_reasons.append("No feature match → Using default columns (Features, Price, Rating)")

    #  Display matches
    print("\n Closest matched features:")
    if found_matches:
        for w, m, c, sc in found_matches:
            print(f"   ➤ '{w}' → '{m}'  → Feature: {c}  [Score: {sc}]")
    else:
        print("   ➤ None (showing general search results)")

    return list(set(selected_columns)), match_reasons


# -----------------------------------------------
#  Feature-Aware AI Search
# -----------------------------------------------
def ai_mobile_search(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    feature_columns, match_reasons = detect_feature_intent(query, keyword_mapping)

    # Weight setup
    column_weights = {col: 1.0 for col in feature_columns}
    default_weight = 0.3
    for col in df.columns:
        if col not in column_weights and df[col].dtype == "object":
            column_weights[col] = default_weight

    # Feature similarity
    total_score = torch.zeros(len(df))
    for col, weight in column_weights.items():
        if col in df.columns and df[col].dtype == "object":
            col_embeddings = embedder.encode(df[col].fillna("").tolist(), convert_to_tensor=True)
            sim_scores = util.cos_sim(query_embedding, col_embeddings)[0]
            total_score += sim_scores.cpu() * weight

    df["Feature_Similarity_Score"] = total_score.numpy()

    # Hybrid ranking
    df["Hybrid_Score"] = (
        (df["Feature_Similarity_Score"] * 0.6) +
        (df["Sentiment_Score"] * 0.2) +
        (df["Rating_out_of_5"] / 5 * 0.2)
    )

    results = df.sort_values(by="Hybrid_Score", ascending=False).head(top_k).reset_index(drop=True)
    return results, match_reasons


# -----------------------------------------------
#  Explain Match Reason (Text + Sentiment)
# -----------------------------------------------
def explain_match(row, query):
    q = query.lower()

    def combine(feature_name, value):
        sentiment = row.get("Sentiment", "Neutral")
        score = round(row.get("Sentiment_Score", 0.0), 2)
        return f"Matched by {feature_name.lower()}: {value} | Sentiment: {sentiment} ({score})"

    if any(k in q for k in keyword_mapping["Camera Quality"]):
        return combine("Camera", row.get("Camera Quality", "N/A"))

    elif any(k in q for k in keyword_mapping["Battery Quality"]):
        return combine("Battery", row.get("Battery Quality", "N/A"))

    elif any(k in q for k in keyword_mapping["Display"]):
        return combine("Display", row.get("Display", "N/A"))

    elif any(k in q for k in keyword_mapping["Performance"]):
        return combine("Performance", row.get("Performance", "N/A"))

    elif any(k in q for k in keyword_mapping["Price_INR"]):
        return combine("Price", f"₹{row.get('Price_INR', 'N/A')}")

    else:
        return combine("Features", row.get("Features", "N/A"))


# -----------------------------------------------
#  Run Example Query
# -----------------------------------------------
if __name__ == "__main__":
    query = input(" Enter your search query: ")
    print("\n Searching feature-aware AI matches...\n")

    results, reasons = ai_mobile_search(query, top_k=5)
    results["Match_Reason"] = results.apply(lambda r: explain_match(r, query), axis=1)

    print("\n Match reasons:")
    for r in reasons:
        print("  -", r)

    print("\n Top Recommended Phones:\n")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['Product Name']} — Score: {round(row['Hybrid_Score'], 3)}")
        print(f"   ➤ {row['Match_Reason']}")
        print()

    results[["Product Name", "Hybrid_Score", "Match_Reason"]].to_csv("ai_mobile_search_results_filtered.csv", index=False)
    print(" Results saved to ai_mobile_search_results_filtered.csv")
