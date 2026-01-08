import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def create_lineplot():
    # Read the exported CSV
    df = pd.read_csv("daily_rentals_2005.csv")

    # Ensure rental_day is datetime
    df["rental_day"] = pd.to_datetime(df["rental_day"])

    # Plot
    plt.figure(figsize=(14, 6))

    for store_id, group in df.groupby("store_id"):
        plt.plot(group["rental_day"], group["rentals"], label=f"Store {store_id}")

    plt.title("Daily Rentals by Store (2005)")
    plt.xlabel("Date")
    plt.ylabel("Number of Rentals")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return plt

def create_barplot():
    # Load the CSV created earlier
    df = pd.read_csv("store_total_benefit.csv")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(df["store_id"], df["total_benefit"])

    plt.title("Total Benefit by Store")
    plt.xlabel("Store ID")
    plt.ylabel("Total Benefit")
    plt.xticks(df["store_id"])   # Show store numbers on the x-axis
    plt.tight_layout()
    plt.show()
    return plt


def create_dataframe():
    # Read the CSV
    df = pd.read_csv("top5_movies_by_store_2005.csv")
    return df
    




class MovieRecommender:
    def __init__(self, csv_path="movies.csv", model_name="all-MiniLM-L6-v2"):
        # Load movie data
        self.movies_df = pd.read_csv(csv_path)
        self.movies_df["description"] = self.movies_df["description"].fillna("")

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings
        self.movie_embeddings = self.model.encode(
            self.movies_df["description"].tolist(), convert_to_tensor=False
        )

    def get_similar_movies(self, user_description, top_n=3):
        if not user_description.strip():
            return []

        # Compute embedding for user input
        user_embedding = self.model.encode([user_description], convert_to_tensor=False)

        # Compute cosine similarity
        similarities = cosine_similarity(user_embedding, self.movie_embeddings)[0]

        # Get top N indices
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Return list of dicts with title, rating, similarity
        results = []
        for idx in top_indices:
            results.append({
                "title": self.movies_df.iloc[idx]["title"],
                "rating": self.movies_df.iloc[idx]["rating"],
                "similarity": similarities[idx]
            })
        return results
