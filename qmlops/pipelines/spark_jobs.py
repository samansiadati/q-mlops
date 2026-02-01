# src/data_pipeline/spark_jobs.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os
import mlflow

def run_metrics_job(raw_dir: str, delta_root: str, run_id: str):
    """
    Reads raw JSON metrics from raw_dir, adds run_id, writes to Delta, and logs raw files to MLflow.
    
    Args:
        raw_dir (str): Path to raw JSON files
        delta_root (str): Root path for Delta tables
        run_id (str): Unique ID for this run
    """
    # ---------- Spark session ----------
    spark = (
        SparkSession.builder
        .appName("QGM-Metrics-Job")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    # ---------- Read raw JSON metrics ----------
    df = spark.read.json(f"{raw_dir}/*.json")

    # ---------- Add run_id column ----------
    df = df.withColumn("run_id", F.lit(run_id))

    # ---------- Write to Delta ----------
    metrics_path = os.path.join(delta_root, "metrics")
    df.write.format("delta").mode("append").save(metrics_path)
    print(f"✅ Metrics written to Delta table at {metrics_path}")

    # ---------- Log raw JSON files to MLflow ----------
    raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".json")]
    for f in raw_files:
        mlflow.log_artifact(f, artifact_path=f"raw/{run_id}")

    # ---------- Stop Spark ----------
    spark.stop()
