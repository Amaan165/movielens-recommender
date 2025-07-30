#!/usr/bin/env python
# -- coding: utf-8 --
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import logging
import numpy as np


def group_by_user(df):
    return df.groupby("userId")["movieId"] \
             .apply(list, meta=('movieId', 'object')) \
             .compute().reset_index().rename(columns={"movieId": "trueItems"})
def dcg(relevance):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))


def evaluate_ndcg(beta, grouped_data, movie_stats, k):
    df = movie_stats.copy()
    df["popularity_score"] = df["rating_sum"] / (df["rating_count"] + beta)
    top_k_list = list(df.sort_values("popularity_score", ascending=False).head(k)["movieId"])
    ndcg_sum = 0
    valid_users = 0
    for row in grouped_data.itertuples():
        true_set = set(row.trueItems)
        if not true_set:
            continue

        # Relevance: 1 if in true set, else 0
        relevance = [1 if mid in true_set else 0 for mid in top_k_list]
        ideal_relevance = sorted(relevance, reverse=True)

        dcg_val = dcg(relevance)
        idcg_val = dcg(ideal_relevance)
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0
        ndcg_sum += ndcg
        valid_users += 1
    return {
        "beta": beta,
        f"ndcg@{k}": ndcg_sum / valid_users if valid_users else 0
    }
def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(_name_)
    workers = args.workers
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"🚀 Starting local Dask cluster with {workers} workers")
    cluster = LocalCluster(n_workers=workers, threads_per_worker=1, memory_limit=args.memory)
    client = Client(cluster)

    # Load data
    val_df = dd.read_parquet(args.val)
    test_df = dd.read_parquet(args.test)
    logger.info(f"📊 Loading precomputed movie statistics from {args.movie_stats}")
    movie_stats = dd.read_parquet(args.movie_stats).compute()
    val_grouped = group_by_user(val_df)
    test_grouped = group_by_user(test_df)

    logger.info(f"🔍 Evaluating NDCG@{args.top_k} over beta ∈ [1, {args.beta_range}]...")
    val_metrics = []
    for beta in range(250, args.beta_range + 1):
        result = evaluate_ndcg(beta, val_grouped, movie_stats, k=args.top_k)
        val_metrics.append(result)
        logger.info(f"📈 Beta={beta:3} → NDCG@{args.top_k}: {result[f'ndcg@{args.top_k}']:.5f}")

    val_metrics_df = pd.DataFrame(val_metrics)
    val_metrics_df.to_csv(os.path.join(args.output_dir, "val_metrics.csv"), index=False)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_metrics_df["beta"], val_metrics_df[f"ndcg@{args.top_k}"])
    plt.xlabel("Beta")
    plt.ylabel(f"NDCG@{args.top_k}")
    plt.title(f"Validation NDCG@{args.top_k} vs Beta")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "validation_ndcg_vs_beta.png"))
    best_beta = val_metrics_df.loc[val_metrics_df[f"ndcg@{args.top_k}"].idxmax(), "beta"]
    logger.info(f"🏁 Best beta = {best_beta}")

    logger.info("🧪 Evaluating on test set...")
    test_result = evaluate_ndcg(best_beta, test_grouped, movie_stats, k=args.top_k)
    pd.DataFrame([{
        "best_beta": best_beta,
        f"test_ndcg@{args.top_k}": test_result[f"ndcg@{args.top_k}"]
    }]).to_csv(os.path.join(args.output_dir, "test_metrics.csv"), index=False)
    print(f"✅ Finished! Best β = {best_beta}, Test NDCG@{args.top_k} = {test_result[f'ndcg@{args.top_k}']:.5f}")
    client.close()
if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Popularity-based Recommender with NDCG@K using Dask")
    parser.add_argument("--val", required=True, help="Path to val.parquet")
    parser.add_argument("--test", required=True, help="Path to test.parquet")
    parser.add_argument("--movie-stats", required=True, help="Path to movie_stats.parquet")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to store results")
    parser.add_argument("--workers", type=int, default=4, help="Number of Dask workers")
    parser.add_argument("--threads", type=int, default=1, help="Threads per worker")
    parser.add_argument("--memory", type=str, default="2GB", help="Memory per worker (e.g., 2GB)")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top items to recommend")
    parser.add_argument("--beta-range", type=int, default=100, help="Maximum beta value to tune")
    args = parser.parse_args()
    main(args)
