import pandas as pd
import random
from scipy.stats import pearsonr
from tqdm import tqdm

# === Load ratings ===
ratings = pd.read_csv("C:\\Users\\shrav\\Downloads\\ml-latest\\ml-latest\\ratings.csv")  # columns: userId, movieId, rating

# === Load user → movie list mapping from Parquet ===
df = pd.read_parquet("grouped_user_movies.parquet")
user_movie_map = df.set_index("userId")["movie_ids"].to_dict()

# === Compute number of ratings per user ===
rating_counts = ratings['userId'].value_counts()

# === Function to compute average correlation for users with > min_ratings ===
def compute_avg_corr(min_ratings=10):
    # Filter user_ids with sufficient number of ratings
    eligible_users = [user for user in user_movie_map if rating_counts.get(user, 0) > min_ratings]
    print(f"🎯 Users with > {min_ratings} ratings: {len(eligible_users)}")

    # Sample 100 valid pairs (no repeats)
    seen = set()
    pairs = []
    while len(pairs) < 100 and len(seen) < (len(eligible_users) * (len(eligible_users) - 1)) // 2:
        u1, u2 = random.sample(eligible_users, 2)
        pair = tuple(sorted((u1, u2)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    # Function to get common ratings
    def get_common_ratings(user1, user2):
        r1 = ratings[ratings['userId'] == user1][['movieId', 'rating']]
        r2 = ratings[ratings['userId'] == user2][['movieId', 'rating']]
        merged = pd.merge(r1, r2, on='movieId', suffixes=('_1', '_2'))
        return merged['rating_1'], merged['rating_2']
    


    # Compute correlations
    correlations = []
    for u1, u2 in tqdm(pairs, desc=f"Computing correlations for >{min_ratings} ratings"):
        r1, r2 = get_common_ratings(u1, u2)
        if len(r1) > 1:
            if r1.nunique() == 1:
                print(f"⚠️ Constant ratings by User {u1}: {r1.tolist()}")
                continue
            if r2.nunique() == 1:
                print(f"⚠️ Constant ratings by User {u2}: {r2.tolist()}")
                continue

            corr, _ = pearsonr(r1, r2)
            correlations.append(corr)

            if corr == 1.0:
                print(f"✅ Identical ratings by Users {u1} and {u2}")
                print(f"Ratings: User {u1} -> {r1.tolist()} | User {u2} -> {r2.tolist()}")

    # Output result
    if correlations:
        avg_corr = sum(correlations) / len(correlations)
        print(f"✅ Average Pearson correlation (> {min_ratings} ratings): {avg_corr:.4f}")
        return avg_corr
    else:
        print("⚠️ Not enough common ratings to compute correlations.")
        return 0

# === Run for multiple thresholds ===
corrs  = []
for i in tqdm(range(20)):
    c = compute_avg_corr(min_ratings=1)
    corrs.append(c)

print("Final corr : ",sum(corrs)/20)
