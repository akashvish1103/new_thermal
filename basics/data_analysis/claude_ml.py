# =============================================================
#  THERMAL FACE ROI — Full ML Pipeline
#  Converts per-frame CSVs → feature matrix → ML analysis
# =============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION
# ─────────────────────────────────────────────

DATA_FOLDER = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\data"          # ← change this to your folder path
OUTPUT_CSV  = "ml_ready.csv"  # final feature file

ROI_COLS = [
    "left_eye_temp",
    "right_eye_temp",
    "forehead_temp",
    "nose_temp",
    "left_cheek_temp",
    "right_cheek_temp",
]


# ─────────────────────────────────────────────
# STEP 2: FEATURE EXTRACTION FUNCTION
# Collapses 2848 rows → 1 row per subject
# ─────────────────────────────────────────────

def extract_features(df, subject_name):
    """
    Given a dataframe of one subject's session,
    returns a dict of ~35 statistical features.
    """
    row = {"subject": subject_name}

    for col in ROI_COLS:
        series = df[col].dropna()
        frames  = df.loc[series.index, "frame"]

        row[f"{col}_mean"]  = series.mean()
        row[f"{col}_std"]   = series.std()
        row[f"{col}_min"]   = series.min()
        row[f"{col}_max"]   = series.max()
        row[f"{col}_range"] = series.max() - series.min()

        # Linear slope across frames (positive = warming, negative = cooling)
        if len(frames) > 1:
            slope = np.polyfit(frames, series, 1)[0]
        else:
            slope = 0.0
        row[f"{col}_slope"] = slope

    # ── Cross-ROI / physiological features ──
    row["eye_asymmetry"]       = row["left_eye_temp_mean"]   - row["right_eye_temp_mean"]
    row["cheek_asymmetry"]     = row["left_cheek_temp_mean"] - row["right_cheek_temp_mean"]
    row["nose_forehead_ratio"] = (
        row["nose_temp_mean"] / row["forehead_temp_mean"]
        if row["forehead_temp_mean"] != 0 else np.nan
    )
    # Stress indicator: nasal tip temp relative to eye temp
    row["nose_eye_delta"]      = row["nose_temp_mean"] - row["left_eye_temp_mean"]

    return row


# ─────────────────────────────────────────────
# STEP 3: LOAD ALL CSVs → BUILD FEATURE MATRIX
# ─────────────────────────────────────────────

files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not files:
    raise FileNotFoundError(f"No CSVs found in '{DATA_FOLDER}'. Check your DATA_FOLDER path.")

all_rows = []

for f in files:
    basename = os.path.basename(f)                  # e.g. aditi_roi_temperatures.csv
    subject  = basename.split("_roi")[0]            # e.g. aditi

    df = pd.read_csv(f)

    # Basic validation
    missing = [c for c in ROI_COLS if c not in df.columns]
    if missing:
        print(f"  [SKIP] {basename} is missing columns: {missing}")
        continue

    row = extract_features(df, subject)
    all_rows.append(row)
    print(f"  [OK] {subject:20s} — {len(df)} frames processed")

final_df = pd.DataFrame(all_rows)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved feature matrix → '{OUTPUT_CSV}'")
print(f"   Shape: {final_df.shape[0]} subjects × {final_df.shape[1]-1} features\n")
print(final_df.set_index("subject").round(4))


# ─────────────────────────────────────────────
# STEP 4: ML ANALYSIS
# (works even with only 10 subjects)
# ─────────────────────────────────────────────

# Separate features from subject labels
X_raw   = final_df.drop(columns=["subject"])
subjects = final_df["subject"].values

# Drop any columns with NaN (e.g. if a file had all-NaN ROI)
X_raw = X_raw.dropna(axis=1)

# Scale features to zero mean, unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

feature_names = X_raw.columns.tolist()


# ── 4a. PCA — Reduce to 2D for visualization ──────────────

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100
print(f"\nPCA: PC1 explains {explained[0]:.1f}%, PC2 explains {explained[1]:.1f}%")

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

for i, (x, y, name) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], subjects)):
    ax.scatter(x, y, color=colors[i], s=120, zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points",
                xytext=(8, 4), fontsize=9)

ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
ax.set_title("PCA of Thermal ROI Features — Each Point = 1 Subject")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_plot.png", dpi=150)
plt.show()
print("📊 Saved: pca_plot.png")


# ── 4b. Correlation Heatmap of Features ───────────────────

fig, ax = plt.subplots(figsize=(14, 10))
corr = X_raw.corr()
sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
            annot=False, linewidths=0.3)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
print("📊 Saved: correlation_heatmap.png")


# ── 4c. K-Means Clustering (if ≥ 3 subjects) ─────────────

n = len(subjects)
if n >= 3:
    best_k, best_score = 2, -1
    k_range = range(2, min(n, 6))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    print(f"\nBest K-Means k={best_k} (silhouette score: {best_score:.3f})")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_colors = plt.cm.Set1(np.linspace(0, 0.8, best_k))

    for i, (x, y, name) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], subjects)):
        c = cluster_labels[i]
        ax.scatter(x, y, color=cluster_colors[c], s=140,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title(f"K-Means Clustering (k={best_k}) on Thermal ROI Features")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_plot.png", dpi=150)
    plt.show()
    print("📊 Saved: kmeans_plot.png")

    # Print cluster assignments
    print("\nCluster Assignments:")
    for name, cluster in zip(subjects, cluster_labels):
        print(f"  {name:20s} → Cluster {cluster + 1}")


# ── 4d. Per-ROI Mean Bar Chart across subjects ────────────

mean_cols = [c for c in feature_names if c.endswith("_mean")]
mean_df   = final_df[["subject"] + mean_cols].set_index("subject")
mean_df.columns = [c.replace("_temp_mean", "").replace("_", " ") for c in mean_df.columns]

fig, ax = plt.subplots(figsize=(12, 5))
mean_df.T.plot(kind="bar", ax=ax, width=0.7)
ax.set_title("Mean ROI Temperature per Subject")
ax.set_xlabel("ROI Region")
ax.set_ylabel("Temperature (°C)")
ax.legend(title="Subject", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("roi_means_bar.png", dpi=150)
plt.show()
print("📊 Saved: roi_means_bar.png")

print("\n✅ All done! Files generated:")
print("   ml_ready.csv           ← your feature matrix")
print("   pca_plot.png           ← subject spread in 2D")
print("   correlation_heatmap.png← feature correlations")
print("   kmeans_plot.png        ← clusters (if ≥3 subjects)")
print("   roi_means_bar.png      ← per-ROI means by subject")