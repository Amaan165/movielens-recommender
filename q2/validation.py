import pandas as pd

# === Step 1: Read the top 100 pairs ===
top_pairs = pd.read_csv("top_100_user_pairs_min_ratings_50.csv")

# === Step 2: Load user→movie mappings from grouped Parquet ===
grouped_df = pd.read_parquet("grouped_user_movies.parquet")
user_movie_map = grouped_df.set_index("userId")["movie_ids"].to_dict()

# === Step 3: Load full ratings CSV to validate existence ===
ratings_df = pd.read_csv("C:\\Users\\shrav\\Downloads\\ml-latest\\ml-latest\\ratings.csv")
valid_user_ids = set(ratings_df["userId"].unique())

# === Step 4: For each user pair, fetch movies, compute Jaccard index, and validate existence ===
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0

for _, row in top_pairs.iterrows():
    uid1, uid2 = row["userId1"], row["userId2"]

    # Check existence in ratings.csv
    exists_1 = uid1 in valid_user_ids
    exists_2 = uid2 in valid_user_ids

    print(f"\nUser {uid1} in CSV: {exists_1}, User {uid2} in CSV: {exists_2}")

    # Get watched movies
    movies1 = set(user_movie_map.get(uid1, []))
    movies2 = set(user_movie_map.get(uid2, []))

    print(f"User {uid1} watched {len(movies1)} movies: {list(movies1)[:10]}")
    print(f"User {uid2} watched {len(movies2)} movies: {list(movies2)[:10]}")

    # Check if movie sets are exactly the same
    if movies1 == movies2:
        print("✅ The users have watched exactly the same set of movies.")
    else:
        print("❌ The users have different movie sets.")

    # Compute Jaccard index
    jaccard_score = jaccard(movies1, movies2)
    print(f"Jaccard Index: {jaccard_score:.4f}")
    print("=================================================================================================================")