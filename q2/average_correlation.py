import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

# Load ratings
ratings = pd.read_csv('C:\\Users\\shrav\\Downloads\\ml-latest\\ml-latest\\ratings.csv')

# Load top 100 similar user pairs
pairs_df = pd.read_csv('top_100_user_pairs_min_ratings_50.csv')  # assumes 'userId1', 'userId2'

def get_common_ratings(user1, user2, ratings_df):
    """Return two aligned lists of ratings for movies both users rated."""
    u1_ratings = ratings_df[ratings_df['userId'] == user1][['movieId', 'rating']]
    u2_ratings = ratings_df[ratings_df['userId'] == user2][['movieId', 'rating']]
    merged = pd.merge(u1_ratings, u2_ratings, on='movieId', suffixes=('_1', '_2'))
    return merged['rating_1'], merged['rating_2'], merged['movieId']

def avg_pairwise_corr(pairs, ratings_df):
    corrs = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        user1, user2 = row['userId1'], row['userId2']
        r1, r2, movie_ids = get_common_ratings(user1, user2, ratings_df)

        if len(r1) > 1:
            if r1.nunique() == 1:
                print(f"⚠️ Constant ratings by User {user1}: {r1.tolist()} on movies {movie_ids.tolist()}")
                continue
            if r2.nunique() == 1:
                print(f"⚠️ Constant ratings by User {user2}: {r2.tolist()} on movies {movie_ids.tolist()}")
                continue

            corr, _ = pearsonr(r1, r2)
            corrs.append(corr)

            if corr == 1.0:
                print(f"✅ Identical ratings by Users {user1} and {user2}")
                print(f"Movies: {movie_ids.tolist()}")
                print(f"Ratings: User {user1} -> {r1.tolist()} | User {user2} -> {r2.tolist()}")
    return sum(corrs) / len(corrs) if corrs else None

# Compute average correlation for top 100 similar pairs
avg_corr_similar = avg_pairwise_corr(pairs_df, ratings)
print(f"📈 Average correlation of similar user pairs: {avg_corr_similar:.4f}")
