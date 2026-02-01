# src/experiments/run_experiment.py
import argparse
import os
import uuid
import datetime
import numpy as np

import mlflow
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable

# Headless plotting
import matplotlib
matplotlib.use("Agg")

from qmlops.quantum.quantum_data_gen import QGMConfig, generate_batches
from qmlops.quantum.noise_models import simple_noise_model
from qmlops.utils.io import ensure_dirs, save_json
from qmlops.utils.mlflow_utils import start_run, log_params, log_metrics, set_tags
from qmlops.utils.plotting import plot_probs
from qmlops.utils.quantum_utils import ensure_hermitian

# ---------------- Spark metrics job ----------------
def run_spark_metrics(input_dir: str, delta_root: str, run_id: str):
    try:
        from qmlops.pipelines.spark_jobs import run_metrics_job
        run_metrics_job(input_dir, delta_root, run_id)
    except ModuleNotFoundError:
        print("⚠️ No data_pipeline found. Skipping Spark metrics job.")

def build_spark(app_name: str):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

def get_timestamped_run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="QGM-Benchmark")
    parser.add_argument("--model-type", choices=["vqc", "qgan", "qgan_entangled"], default="qgan_entangled")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--backend", default="aer_simulator")
    parser.add_argument("--entangle", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--delta-root", default="data/delta")
    parser.add_argument("--clean", action="store_true", help="Remove previous runs/data")
    args = parser.parse_args()

    if args.clean:
        import shutil
        for folder in [args.raw_dir, args.delta_root, "data/figures", "artifacts", "mlruns"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"🧹 Removed folder: {folder}")

    ensure_dirs(args.raw_dir, args.delta_root, "data/figures", "artifacts")

    # ---------- Run setup ----------
    run_id = get_timestamped_run_id()
    cfg = QGMConfig(
        qubits=args.qubits,
        depth=args.depth,
        shots=args.shots,
        entangle=args.entangle,
        seed=args.seed,
        backend=args.backend,
        noise_model=simple_noise_model(),
        model_type=args.model_type,
    )

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    tags = {
        "model_type": args.model_type,
        "backend": args.backend,
        "noise_model": "simple_noise_model",
        "orchestrator": "spark+delta",
        "paper_pipeline": "Qiskit→Spark→Delta→MLflow",
        "run_group": run_id,
    }

    spark = None

    with start_run(args.experiment_name, tags):
        try:
            log_params({
                "circuit_depth": args.depth,
                "shots": args.shots,
                "qubits": args.qubits,
                "epochs": args.epochs,
                "entangle": args.entangle,
                "backend": args.backend,
                "seed": args.seed,
                "model_type": args.model_type,
            })

            # --- Generate raw quantum data ---
            raw_run_dir = os.path.join(args.raw_dir, run_id)
            ensure_dirs(raw_run_dir)
            generate_batches(raw_run_dir, run_id, args.epochs, cfg)

            # --- Spark aggregation & Delta write ---
            run_spark_metrics(raw_run_dir, args.delta_root, run_id)

            # --- Spark readback ---
            spark = build_spark("QGMReadback")
            metrics_path = os.path.join(args.delta_root, "metrics")

            df = (
                spark.read.format("delta")
                .load(metrics_path)
                .filter(f"run_id = '{run_id}'")
                .orderBy("epoch")
            )

            rows = df.collect()
            if not rows:
                raise RuntimeError(f"No metrics rows found for run_id={run_id}")

            for r in rows:
                epoch = int(getattr(r, "epoch", 0))
                probs = getattr(r, "probs", None)
                ref = getattr(r, "ref", None)
                fidelity = float(getattr(r, "fidelity", np.nan))
                kl = float(getattr(r, "kl", np.nan))
                jsd = float(getattr(r, "jsd", np.nan))
                trace = float(getattr(r, "trace", np.nan))

                if probs is None or not isinstance(probs, (list, np.ndarray)):
                    print(f"⚠️ Epoch {epoch}: 'probs' invalid, replacing with identity matrix")
                    probs = np.eye(cfg.qubits)
                else:
                    probs = np.asarray(probs)
                    if probs.ndim != 2 or probs.shape[0] != cfg.qubits:
                        print(f"⚠️ Epoch {epoch}: 'probs' shape invalid, replacing with identity matrix")
                        probs = np.eye(cfg.qubits)

                if ref is None or not isinstance(ref, (list, np.ndarray)):
                    print(f"⚠️ Epoch {epoch}: 'ref' invalid, replacing with uniform vector")
                    ref = np.ones(cfg.qubits) / cfg.qubits
                else:
                    ref = np.asarray(ref)

                probs = ensure_hermitian(probs)

                log_metrics({
                    "fidelity": fidelity,
                    "kl": kl,
                    "jsd": jsd,
                    "trace": trace,
                }, step=epoch)

                fig_dir = os.path.join("data/figures", run_id)
                ensure_dirs(fig_dir)
                fig_path = os.path.join(fig_dir, f"epoch{epoch:04d}.png")
                plot_probs(probs, ref, f"Run {run_id} · Epoch {epoch}", fig_path)
                mlflow.log_artifact(fig_path, artifact_path=f"figures/{run_id}")

            # --- Save config ---
            artifact_dir = os.path.join("artifacts", run_id)
            ensure_dirs(artifact_dir)
            cfg_dict = cfg.__dict__.copy()
            cfg_dict["noise_model"] = str(cfg_dict["noise_model"])
            save_json(os.path.join(artifact_dir, "config.json"), {"run_id": run_id, "config": cfg_dict})
            mlflow.log_artifacts(artifact_dir)

        except Exception:
            mlflow.set_tag("status", "failed")
            raise
        else:
            mlflow.set_tag("status", "completed")
        finally:
            if spark:
                spark.stop()
            set_tags({"summary": f"Completed {args.epochs} epochs · run_id={run_id}"})
            print(f"✅ Run complete · run_id={run_id}")

if __name__ == "__main__":
    main()
