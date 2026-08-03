# =============================================================
#  THERMAL ROI — Interactive Clustering Explorer
#  Change algorithm and number of clusters freely
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ██  CONFIGURATION  — Change these two lines only
# ─────────────────────────────────────────────────────────────

ALGORITHM    = "gmm"   # Options: "kmeans" | "agglomerative" | "gmm" |
                          #          "spectral" | "birch" | "dbscan"

N_CLUSTERS   = 3          # Number of clusters
                          # (ignored for DBSCAN — it finds clusters automatically)

# DBSCAN-only parameters (only used if ALGORITHM = "dbscan")
DBSCAN_EPS   = 1.5        # neighbourhood radius — increase if too many noise points
DBSCAN_MIN_SAMPLES = 2    # min points to form a core point

# ─────────────────────────────────────────────────────────────

# CSV_PATH = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\ml_ready.csv"   # ← path to your feature file
CSV_PATH = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\ml_ready_excluded.csv"

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD & SCALE
# ─────────────────────────────────────────────────────────────

df       = pd.read_csv(CSV_PATH)
subjects = df["subject"].values
X_raw    = df.drop(columns=["subject"]).dropna(axis=1)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA for 2D visualization (always done regardless of algorithm)
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100


# ─────────────────────────────────────────────────────────────
# STEP 2: FIT THE CHOSEN ALGORITHM
# ─────────────────────────────────────────────────────────────

algo_name = ALGORITHM.lower().strip()

if algo_name == "kmeans":
    model  = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)

elif algo_name == "agglomerative":
    # Hierarchical — try linkage: "ward" | "complete" | "average" | "single"
    model  = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    labels = model.fit_predict(X_scaled)

elif algo_name == "gmm":
    # Gaussian Mixture — probabilistic soft clustering
    model  = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=10)
    model.fit(X_scaled)
    labels = model.predict(X_scaled)

elif algo_name == "spectral":
    # Works well when clusters are non-convex
    model  = SpectralClustering(n_clusters=N_CLUSTERS, random_state=42,
                                affinity="rbf", assign_labels="kmeans")
    labels = model.fit_predict(X_scaled)

elif algo_name == "birch":
    # Good for large datasets; uses a tree structure
    model  = Birch(n_clusters=N_CLUSTERS, threshold=0.5)
    labels = model.fit_predict(X_scaled)

elif algo_name == "dbscan":
    # Density-based — no need to specify k; finds clusters of arbitrary shape
    # Label -1 = noise point (doesn't belong to any cluster)
    model  = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = model.fit_predict(X_scaled)
    N_CLUSTERS = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"DBSCAN found {N_CLUSTERS} cluster(s) + "
          f"{(labels == -1).sum()} noise point(s)")

else:
    raise ValueError(f"Unknown algorithm '{ALGORITHM}'. "
                     "Choose: kmeans | agglomerative | gmm | spectral | birch | dbscan")


# ─────────────────────────────────────────────────────────────
# STEP 3: EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

unique_labels = set(labels)
non_noise     = labels != -1   # for DBSCAN noise handling

print(f"\n{'='*55}")
print(f"  Algorithm : {ALGORITHM.upper()}")
print(f"  Clusters  : {N_CLUSTERS}")
print(f"{'='*55}")

# Metrics need at least 2 clusters and no all-noise scenario
if len(unique_labels - {-1}) >= 2 and non_noise.sum() > 0:
    sil = silhouette_score(X_scaled[non_noise], labels[non_noise])
    db  = davies_bouldin_score(X_scaled[non_noise], labels[non_noise])
    ch  = calinski_harabasz_score(X_scaled[non_noise], labels[non_noise])
    print(f"  Silhouette Score      : {sil:.3f}  (higher = better, max 1.0)")
    print(f"  Davies-Bouldin Score  : {db:.3f}   (lower  = better, min 0.0)")
    print(f"  Calinski-Harabasz     : {ch:.1f} (higher = better)")
else:
    print("  ⚠ Not enough clusters for metrics (try adjusting parameters)")

print(f"\n  Cluster Assignments:")
for name, lbl in sorted(zip(subjects, labels), key=lambda x: x[1]):
    tag = f"Cluster {lbl+1}" if lbl != -1 else "NOISE (outlier)"
    print(f"    {name:20s} → {tag}")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# STEP 4: PLOTS
# ─────────────────────────────────────────────────────────────

# Color palette — handles noise (-1) as grey
unique_sorted = sorted(unique_labels)
palette       = plt.cm.Set1(np.linspace(0, 0.9, max(len(unique_sorted), 2)))
color_map     = {lbl: ("lightgrey" if lbl == -1 else palette[i])
                 for i, lbl in enumerate(unique_sorted)}

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f"Thermal ROI Clustering  |  Algorithm: {ALGORITHM.upper()}  |  k={N_CLUSTERS}",
    fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)


# ── Plot 1: PCA scatter coloured by cluster ──────────────────
ax1 = fig.add_subplot(gs[0, 0])

for lbl in unique_sorted:
    mask = labels == lbl
    tag  = f"Cluster {lbl+1}" if lbl != -1 else "Noise"
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=color_map[lbl], s=150,
                edgecolors="black", linewidths=0.5,
                label=tag, zorder=3)

for x, y, name in zip(X_pca[:, 0], X_pca[:, 1], subjects):
    ax1.annotate(name, (x, y), textcoords="offset points",
                 xytext=(7, 4), fontsize=8)

ax1.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
ax1.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
ax1.set_title("PCA — Cluster View")
ax1.legend(fontsize=8, loc="best")
ax1.grid(True, alpha=0.3)


# ── Plot 2: Silhouette sweep — best k for this algorithm ─────
ax2 = fig.add_subplot(gs[0, 1])

if algo_name != "dbscan":
    k_range = range(2, min(len(subjects), 8))
    sil_scores = []

    for k in k_range:
        if algo_name == "kmeans":
            m = KMeans(n_clusters=k, random_state=42, n_init=20)
        elif algo_name == "agglomerative":
            m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        elif algo_name == "gmm":
            m = GaussianMixture(n_components=k, random_state=42, n_init=10)
        elif algo_name == "spectral":
            m = SpectralClustering(n_clusters=k, random_state=42,
                                   affinity="rbf", assign_labels="kmeans")
        elif algo_name == "birch":
            m = Birch(n_clusters=k, threshold=0.5)

        if algo_name == "gmm":
            m.fit(X_scaled); lbl_k = m.predict(X_scaled)
        else:
            lbl_k = m.fit_predict(X_scaled)

        try:
            sil_scores.append(silhouette_score(X_scaled, lbl_k))
        except Exception:
            sil_scores.append(0)

    bars = ax2.bar(k_range, sil_scores,
                   color=["tomato" if k == N_CLUSTERS else "steelblue"
                          for k in k_range],
                   edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Silhouette vs k  ({ALGORITHM.upper()})\nRed = your current k")
    ax2.set_xticks(list(k_range))
    ax2.grid(axis="y", alpha=0.3)

    best_k = list(k_range)[np.argmax(sil_scores)]
    ax2.annotate(f"Best k={best_k}", xy=(best_k, max(sil_scores)),
                 xytext=(best_k + 0.3, max(sil_scores) - 0.03),
                 fontsize=8, color="green",
                 arrowprops=dict(arrowstyle="->", color="green"))
else:
    ax2.text(0.5, 0.5, "DBSCAN auto-determines\nnumber of clusters\n(no k sweep needed)",
             ha="center", va="center", fontsize=11, transform=ax2.transAxes)
    ax2.set_title("k Sweep (N/A for DBSCAN)")
    ax2.axis("off")


# ── Plot 3: Mean ROI temperatures per cluster ────────────────
ax3 = fig.add_subplot(gs[1, 0])

mean_cols = [c for c in X_raw.columns if c.endswith("_mean")]
roi_labels = [c.replace("_temp_mean", "").replace("_", " ") for c in mean_cols]

cluster_means = {}
for lbl in unique_sorted:
    if lbl == -1:
        continue
    mask = labels == lbl
    cluster_means[f"Cluster {lbl+1}"] = X_raw.loc[mask, mean_cols].mean().values

cm_df = pd.DataFrame(cluster_means, index=roi_labels)
cm_df.plot(kind="bar", ax=ax3, width=0.7,
           color=[color_map[lbl] for lbl in unique_sorted if lbl != -1])
ax3.set_title("Mean ROI Temperature per Cluster")
ax3.set_ylabel("Temperature (°C)")
ax3.set_xlabel("")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=8)


# ── Plot 4: Feature importance — what drives the clusters ────
ax4 = fig.add_subplot(gs[1, 1])

# Use PCA loadings to show which features contribute most to PC1
loadings   = pd.Series(np.abs(pca.components_[0]), index=X_raw.columns)
top10      = loadings.nlargest(10)
short_names = [n.replace("_temp", "").replace("_", " ") for n in top10.index]

ax4.barh(short_names[::-1], top10.values[::-1],
         color="mediumslateblue", edgecolor="black", linewidth=0.5)
ax4.set_xlabel("Absolute Loading on PC1")
ax4.set_title("Top 10 Features Driving PC1\n(main separation axis)")
ax4.grid(axis="x", alpha=0.3)

plt.savefig("clustering_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Saved: clustering_result.png")
print("\n✅ Done! To try a different setup, change ALGORITHM and N_CLUSTERS at the top.")