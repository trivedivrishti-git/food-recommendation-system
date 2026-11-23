import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_and_save_model(csv_path="food_dataset.csv", model_path="model.pkl"):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df["ingredients"] = df["ingredients"].fillna("")
    df["diet"] = df["diet"].fillna("")
    df["course"] = df["course"].fillna("")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

    df["tags"] = (df["ingredients"] + " " + df["diet"] + " " + df["course"]).str.lower()

    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    print("Computing Cosine Similarity Matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    model_data = {
        "df": df,
        "tfidf_vectorizer": tfidf,
        "similarity_matrix": similarity_matrix,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model successfully trained and saved to '{model_path}'.")

if __name__ == "__main__":
    train_and_save_model()