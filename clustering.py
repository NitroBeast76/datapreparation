"""
clustering.py
-------------
All-in-one clustering pipeline for the anime recommendation system.

What this script does:
  1. Loads the scaled feature matrix (anime_features_scaled.csv).
  2. Runs an elbow test on a sample and saves elbow_chart.png.
  3. Recommends a number of clusters (k) based on sqrt(n) rule.
  4. Trains a KMeans model on the full dataset.
  5. Saves the trained model as kmeans_model.pkl.
  6. Merges cluster IDs with anime_clean.csv to create anime_data.csv.
  7. Writes a summary file (clustering_summary.txt).

How to use:
  - Make sure anime_clean.csv and anime_features_scaled.csv are in the same folder.
  - Run: python clustering.py
  - Optional: override k with a command-line argument: python clustering.py --k 150

Outputs:
  - elbow_chart.png           (graph to help choose k)
  - kmeans_model.pkl          (trained KMeans model)
  - anime_data.csv            (clean dataset + cluster_id column)
  - clustering_summary.txt    (summary report)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import os
import sys
import argparse

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
FEATURE_FILE = "anime_features_scaled.csv"
ANIME_CLEAN_FILE = "anime_clean.csv"
OUTPUT_DATA_FILE = "anime_data.csv"
MODEL_FILE = "kmeans_model.pkl"
CHART_FILE = "elbow_chart.png"
SUMMARY_FILE = "clustering_summary.txt"

# Sample size for elbow test (to keep it fast)
ELBOW_SAMPLE_SIZE = 2000

# Default k: sqrt of number of samples (a common heuristic)
# Will be overridden by --k argument if provided.
DEFAULT_K = None  # We'll compute sqrt(n) if not specified

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def load_and_prepare_data():
    """Load feature matrix and clean dataset. Verify alignment."""
    print("\n📂 Loading data...")
    feat_df = pd.read_csv(FEATURE_FILE)
    print(f"   Feature matrix: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} columns")

    if "title" not in feat_df.columns:
        raise KeyError(f"'{FEATURE_FILE}' must contain a 'title' column.")

    anime_df = pd.read_csv(ANIME_CLEAN_FILE)
    print(f"   Clean dataset: {anime_df.shape[0]:,} rows × {anime_df.shape[1]} columns")

    # Sort both by title to ensure alignment
    feat_df = feat_df.sort_values(by="title").reset_index(drop=True)
    anime_df = anime_df.sort_values(by="title").reset_index(drop=True)

    # Verify row counts
    if len(feat_df) != len(anime_df):
        raise ValueError(
            f"Row count mismatch: feature matrix has {len(feat_df)} rows, "
            f"clean dataset has {len(anime_df)} rows."
        )

    # Verify title alignment
    mismatches = (feat_df["title"] != anime_df["title"]).sum()
    if mismatches > 0:
        raise ValueError(
            f"Title alignment failed: {mismatches} titles don't match.\n"
            "Ensure both files contain the same set of anime in the same order."
        )

    print("   ✓ Titles aligned correctly.\n")

    # Separate features (drop title)
    titles = feat_df["title"]
    X = feat_df.drop(columns=["title"]).values

    return X, titles, anime_df


def run_elbow_test(X, sample_size=ELBOW_SAMPLE_SIZE):
    """Run elbow test on a sample and save the plot."""
    print("🔄 Running elbow test (on a sample for speed)...")

    # Sample if dataset is large
    if len(X) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        sample_size = len(X)

    print(f"   Sample size: {sample_size:,}")

    # Test k from 2 to 20
    cluster_range = range(2, 21)
    inertias = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_sample)
        inertias.append(kmeans.inertia_)
        print(f"   k={k:2d}  →  inertia: {kmeans.inertia_:,.0f}")

    # Create elbow plot
    plt.figure(figsize=(10, 5))
    plt.plot(list(cluster_range), inertias, marker='o', linewidth=2, color='steelblue')
    plt.title("Elbow Method — Optimal Number of Clusters", fontsize=14)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia (lower = tighter clusters)", fontsize=12)
    plt.xticks(list(cluster_range))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    print(f"\n   ✓ Elbow chart saved: {CHART_FILE}\n")

    return inertias


def suggest_k(X, user_k=None):
    """
    Determine the number of clusters to use.
    If user_k provided via command line, use that.
    Otherwise, use sqrt(n) as a reasonable default for recommendation systems.
    """
    if user_k is not None:
        k = user_k
        print(f"📌 Using user-specified k = {k}")
    else:
        n = len(X)
        k = int(np.sqrt(n))
        print(f"📌 Using sqrt(n) rule: sqrt({n:,}) ≈ {k}")
    return k


def cluster_full_dataset(X, k):
    """Train KMeans on full dataset and return model + labels."""
    print(f"\n🚀 Training KMeans with k={k} on full dataset ({len(X):,} rows)...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    print(f"   Inertia: {kmeans.inertia_:,.0f}")
    print(f"   Iterations: {kmeans.n_iter_}")
    return kmeans


def save_outputs(kmeans, anime_df, titles, X):
    """Save model, merge data, write summary."""
    # Save model
    joblib.dump(kmeans, MODEL_FILE)
    print(f"\n💾 Model saved: {MODEL_FILE}")

    # Add cluster_id to clean dataset
    anime_df["cluster_id"] = kmeans.labels_
    anime_df["cluster_id"] = anime_df["cluster_id"].astype(int)

    # Reorder columns: cluster_id first, then everything else
    cols = ["cluster_id"] + [c for c in anime_df.columns if c != "cluster_id"]
    anime_df = anime_df[cols]

    # Save final dataset
    anime_df.to_csv(OUTPUT_DATA_FILE, index=False, encoding="utf-8")
    print(f"📁 Final dataset saved: {OUTPUT_DATA_FILE} ({len(anime_df):,} rows)")

    # Cluster size stats
    cluster_counts = anime_df["cluster_id"].value_counts().sort_index()
    min_size = cluster_counts.min()
    max_size = cluster_counts.max()
    median_size = cluster_counts.median()
    mean_size = cluster_counts.mean()

    print("\n📊 Cluster size distribution:")
    print(f"   Min:    {min_size}")
    print(f"   Max:    {max_size}")
    print(f"   Median: {median_size:.0f}")
    print(f"   Mean:   {mean_size:.1f}")

    # Write summary file
    summary_text = f"""Clustering Summary
==================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Number of clusters chosen: {kmeans.n_clusters}
Total anime titles: {len(anime_df):,}
Average titles per cluster: {len(anime_df) / kmeans.n_clusters:.1f}

Cluster size statistics:
  Minimum: {min_size}
  Maximum: {max_size}
  Median:  {median_size:.0f}
  Mean:    {mean_size:.1f}

Files produced:
  - {CHART_FILE}           (elbow method plot)
  - {MODEL_FILE}           (trained KMeans model)
  - {OUTPUT_DATA_FILE}     (clean data with cluster_id)
  - {SUMMARY_FILE}         (this file)

How to use the outputs:
  - Place {OUTPUT_DATA_FILE} in your Flask app directory.
  - Update app.py to use anime_file="{OUTPUT_DATA_FILE}".
  - The model ({MODEL_FILE}) can be loaded for future predictions.
"""
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n📄 Summary saved: {SUMMARY_FILE}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Anime clustering pipeline.")
    parser.add_argument("--k", type=int, help="Number of clusters (overrides automatic selection)")
    args = parser.parse_args()

    print("=" * 70)
    print(" ANIME CLUSTERING PIPELINE")
    print("=" * 70)

    # Step 1: Load data
    X, titles, anime_df = load_and_prepare_data()

    # Step 2: Run elbow test (optional but informative)
    run_elbow_test(X)

    # Step 3: Determine k
    k = suggest_k(X, user_k=args.k)

    # Step 4: Cluster full dataset
    kmeans = cluster_full_dataset(X, k)

    # Step 5: Save all outputs
    save_outputs(kmeans, anime_df, titles, X)

    print("\n" + "=" * 70)
    print(" ✅ ALL DONE!")
    print("=" * 70)
    print(f"Your app‑ready file is: {OUTPUT_DATA_FILE}")
    print("You can now run your Flask recommendation app!")


if __name__ == "__main__":
    main()