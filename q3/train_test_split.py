import pandas as pd
import random

# Load the grouped user-movies parquet
df = pd.read_parquet("../grouped_user_movies.parquet")
user_movie_map = df.set_index("userId")["movie_ids"].to_dict()

# Set minimum number of movies required to keep user
min_movies = 5

# Output lists
train_rows = []
val_rows = []
test_rows = []

# Split ratio
train_ratio = 0.8
val_ratio = 0.1  # test will take the remaining 0.1

for user_id, movie_ids in user_movie_map.items():
    if len(movie_ids) < min_movies:
        continue  # skip users with very few movies

    movies = list(movie_ids)
    random.shuffle(movies)

    n = len(movies)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train = movies[:train_end]
    val = movies[train_end:val_end]
    test = movies[val_end:]

    train_rows.extend([(user_id, mid) for mid in train])
    val_rows.extend([(user_id, mid) for mid in val])
    test_rows.extend([(user_id, mid) for mid in test])

# Convert to DataFrames
train_df = pd.DataFrame(train_rows, columns=["userId", "movieId"])
val_df = pd.DataFrame(val_rows, columns=["userId", "movieId"])
test_df = pd.DataFrame(test_rows, columns=["userId", "movieId"])

# Save to Parquet
train_df.to_parquet("train.parquet", index=False)
val_df.to_parquet("val.parquet", index=False)
test_df.to_parquet("test.parquet", index=False)

# Summary
print(f"Train: {len(train_df)} interactions")
print(f"Val: {len(val_df)} interactions")
print(f"Test: {len(test_df)} interactions")
