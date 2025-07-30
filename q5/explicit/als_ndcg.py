import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import expr

def main(args):
    spark = SparkSession.builder.appName("ALS_Single_Run_NDCG_With_Backfill").getOrCreate()

    print("📥 Loading data...")
    train_df = spark.read.parquet(args.train)
    val_df = spark.read.parquet(args.val)
    test_df = spark.read.parquet(args.test)

    print("🔄 Casting columns...")
    als_train = train_df.selectExpr("cast(userId as int) as user", "cast(movieId as int) as item", "cast(rating as float) as rating")
    als_val = val_df.selectExpr("cast(userId as int) as user", "cast(movieId as int) as item")
    als_test = test_df.selectExpr("cast(userId as int) as user", "cast(movieId as int) as item")

    print("♻️ Backfilling missing movies in val/test...")
    train_movies = als_train.select("item").distinct()
    val_only = als_val.select("item").distinct().join(train_movies, on="item", how="left_anti")
    test_only = als_test.select("item").distinct().join(train_movies, on="item", how="left_anti")

    val_backfill = val_df.join(val_only, val_df.movieId == val_only.item, "inner").drop("item").dropDuplicates(["movieId"])
    test_backfill = test_df.join(test_only, test_df.movieId == test_only.item, "inner").drop("item").dropDuplicates(["movieId"])
    backfill_df = val_backfill.unionByName(test_backfill).selectExpr("cast(userId as int) as user", "cast(movieId as int) as item", "cast(rating as float) as rating")

    als_train = als_train.unionByName(backfill_df)
    print(f"✅ Backfilled {backfill_df.count()} rows into training set.")

    print("⚙️ Training ALS model...")
    als = ALS(
        rank=args.rank,
        maxIter=args.max_iter,
        regParam=args.reg_param,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        coldStartStrategy="drop",
        implicitPrefs=False
    )
    model = als.fit(als_train)

    print(f"📈 Generating top-{args.k} recommendations...")
    user_recs = model.recommendForAllUsers(args.k).withColumn(
        "recommendations", expr("transform(recommendations, x -> cast(x.item as double))")
    )

    print("🧪 Preparing ground truth...")
    val_truth = als_val.groupBy("user").agg(expr("transform(collect_list(item), x -> cast(x as double))").alias("groundTruth"))
    test_truth = als_test.groupBy("user").agg(expr("transform(collect_list(item), x -> cast(x as double))").alias("groundTruth"))
    val_joined = user_recs.join(val_truth, on="user")
    test_joined = user_recs.join(test_truth, on="user")

    print("📐 Evaluating NDCG@K...")
    ranking_eval = RankingEvaluator(predictionCol="recommendations", labelCol="groundTruth", metricName="ndcgAtK", k=args.k)
    val_ndcg = ranking_eval.evaluate(val_joined)
    test_ndcg = ranking_eval.evaluate(test_joined)

    print(f"✅ Validation NDCG@{args.k}: {val_ndcg:.5f}")
    print(f"✅ Test NDCG@{args.k}: {test_ndcg:.5f}")

    file_name = f"als_result_rank{args.rank}_iter{args.max_iter}_reg{args.reg_param}.csv"
    pd.DataFrame([{
        "rank": args.rank,
        "regParam": args.reg_param,
        "maxIter": args.max_iter,
        f"val_ndcg@{args.k}": val_ndcg,
        f"test_ndcg@{args.k}": test_ndcg
    }]).to_csv(file_name, index=False)
    print(f"💾 Results saved to {file_name}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS Single Model Evaluation with Backfill and NDCG@K")
    parser.add_argument("--train", required=True, help="Path to train_enriched.parquet")
    parser.add_argument("--val", required=True, help="Path to val_enriched.parquet")
    parser.add_argument("--test", required=True, help="Path to test_enriched.parquet")
    parser.add_argument("--k", type=int, default=100, help="Top-K recommendations")
    parser.add_argument("--rank", type=int, required=True, help="ALS latent factors")
    parser.add_argument("--max-iter", type=int, required=True, help="Max ALS iterations")
    parser.add_argument("--reg-param", type=float, required=True, help="ALS regularization parameter")
    args = parser.parse_args()
    main(args)
