#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dask pipeline to group movieIds by userId for MinHash preprocessing.
Usage:
    python minhash_group_dask.py --input ratings.parquet --output grouped_user_movies.parquet --workers 4 --scale 4
"""

import argparse
import logging
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

def collect_movies(df):
    """
    Groups movies per user within a partition.
    """
    return df.groupby("userId")["movieId"].apply(list).reset_index(name="movie_ids")

def main(input_path, output_path, workers, scale):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 Starting local Dask cluster with {workers} workers × scale {scale}...")
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
    client = Client(cluster)
    logger.info(f"🔗 Dashboard: {client.dashboard_link}")

    # Read input
    logger.info(f"📥 Reading ratings from: {input_path}")
    df = dd.read_parquet(input_path)

    # Group by userId and collect movieId lists
    logger.info("🔄 Grouping by userId and collecting movieIds...")
    grouped = df.map_partitions(collect_movies)

    logger.info("🔄 Merging userId entries across partitions...")
    grouped = grouped.groupby("userId").agg({"movie_ids": "sum"}).reset_index()

    # ✅ Validate row count matches unique user count
    unique_users = grouped["userId"].nunique().compute()
    row_count = grouped.shape[0].compute()
    logger.info(f"🧪 Unique userId count: {unique_users}")
    logger.info(f"🧮 Row count: {row_count}")
    assert unique_users == row_count, "❌ Mismatch: userId values are not unique!"

    # ✅ Validate movie_ids are lists and non-null (on a sample)
    logger.info("🧪 Validating sample rows...")
    sample = grouped.head(1000)
    for idx, row in sample.iterrows():
        assert isinstance(row["movie_ids"], list), f"❌ Row {idx} movie_ids is not a list"
        assert row["movie_ids"] is not None, f"❌ Row {idx} movie_ids is None"
    logger.info("✅ Sample validation passed.")

    # Save to output
    logger.info(f"💾 Writing grouped user-movie data to: {output_path}")
    grouped.to_parquet(output_path, engine="pyarrow", write_index=False)

    logger.info("✅ Dask MinHash preprocessing complete.")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group movie ratings for MinHash")
    parser.add_argument("--input", required=True, help="Path to input ratings.parquet")
    parser.add_argument("--output", required=True, help="Path to save grouped_user_movies.parquet")
    parser.add_argument("--workers", type=int, default=4, help="Number of Dask workers (default: 4)")
    parser.add_argument("--scale", type=int, default=4, help="Partitions per worker (default: 4)")
    args = parser.parse_args()

    main(args.input, args.output, args.workers, args.scale)
