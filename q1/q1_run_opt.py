#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

# === Step 1: Load grouped user → movie list mapping ===
print("📥 Loading grouped_user_movies.parquet...")
df = pd.read_parquet("grouped_user_movies.parquet")
user_movie_map = df.set_index("userId")["movie_ids"].to_dict()
user_movie_count = {uid: len(movies) for uid, movies in user_movie_map.items()}
print(f"✅ Loaded {len(user_movie_map)} users.")

# === Step 2: Create MinHash signatures ===
print("🔁 Creating MinHash signatures...")
minhashes = {}
num_perm = 128

for user_id in tqdm(user_movie_map, desc="MinHash"):
    m = MinHash(num_perm=num_perm)
    for movie_id in user_movie_map[user_id]:
        m.update(str(movie_id).encode("utf8"))
    minhashes[user_id] = m

# === Step 3: Insert into LSH ===
print("⚡ Indexing users into LSH...")
lsh = MinHashLSH(threshold=0.8, num_perm=num_perm)

for user_id in tqdm(minhashes, desc="LSH Insert"):
    lsh.insert(str(user_id), minhashes[user_id])

# === Step 4: Find similar pairs and estimate Jaccard ===
print("🔍 Querying and scoring candidate pairs...")
results = []

for uid in tqdm(minhashes, desc="Similarity Search"):
    mh = minhashes[uid]
    for other in lsh.query(mh):
        other_id = int(float(other))
        if uid < other_id:
            jaccard = mh.jaccard(minhashes[other_id])
            results.append((uid, other_id, jaccard))

print(f"🧮 Found {len(results)} candidate pairs.")

# === Step 5: Create DataFrame and analyze thresholds ===
df_pairs = pd.DataFrame(results, columns=["userId1", "userId2", "estimated_jaccard"])
df_pairs["user1_ratings"] = df_pairs["userId1"].map(user_movie_count)
df_pairs["user2_ratings"] = df_pairs["userId2"].map(user_movie_count)

thresholds = [5, 10, 20, 30, 40, 50]
print("\n📊 Saving top 100 user pairs for rating thresholds:")

for t in thresholds:
    filtered = df_pairs[(df_pairs["user1_ratings"] > t) & (df_pairs["user2_ratings"] > t)]
    top_100 = filtered.sort_values("estimated_jaccard", ascending=False).head(100)
    filename = f"top_100_user_pairs_min_ratings_{t}.csv"
    top_100.to_csv(filename, index=False)
    print(f" - >{t} ratings: saved {len(top_100)} pairs to {filename}")

print("\n✅ All filtered top-100 pairs saved by rating thresholds.")
