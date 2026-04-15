#!/usr/bin/env python3
"""
Compare benchmark results across datasets (SIFT vs GloVe).
Reads CSVs from each dataset's results/ folder and generates
side-by-side comparison charts in the comparison/ folder.

Usage:
    python compare.py
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_DIR = os.path.join(SCRIPT_DIR, "comparison")
K = 10

DATASETS = {
    "sift": {"label": "SIFT-128d", "color": "#1f77b4", "csv": os.path.join(SCRIPT_DIR, "sift", "results", "benchmark_results.csv")},
    "glove": {"label": "GloVe-200d", "color": "#ff7f0e", "csv": os.path.join(SCRIPT_DIR, "glove", "results", "benchmark_results.csv")},
}


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "n": int(r["n"]),
                "method": r["method"],
                "build_time": float(r["build_time"]),
                "mean_latency": float(r["mean_latency"]),
                "median_latency": float(r["median_latency"]),
                "p99_latency": float(r["p99_latency"]),
                "throughput": float(r["throughput"]),
                "recall": float(r["recall"]),
                "lsh_tables": int(r["lsh_tables"]) if r["lsh_tables"] else None,
                "lsh_bits": int(r["lsh_bits"]) if r["lsh_bits"] else None,
            }
            rows.append(row)
    return rows


def _best_lsh(rows, n):
    lsh = [r for r in rows if r["n"] == n and r["method"].startswith("lsh_")]
    return max(lsh, key=lambda r: r["recall"]) if lsh else None


def _get_method_row(rows, n, method):
    return next((r for r in rows if r["n"] == n and r["method"] == method), None)


def common_sizes(data):
    sets = [set(r["n"] for r in rows) for rows in data.values()]
    return sorted(set.intersection(*sets)) if sets else []


# ---------------------------------------------------------------------------
# Chart 1: HNSW recall comparison across datasets
# ---------------------------------------------------------------------------
def plot_hnsw_recall(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, cfg in DATASETS.items():
        rows = data[name]
        vals = []
        ns = []
        for n in sizes:
            r = _get_method_row(rows, n, "hnsw")
            if r:
                vals.append(r["recall"])
                ns.append(n)
        if vals:
            ax.plot(ns, vals, marker="D", linewidth=2, markersize=8,
                    color=cfg["color"], label=cfg["label"])

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel(f"HNSW Recall@{K}", fontsize=11)
    ax.set_title(f"HNSW Recall@{K} by Dataset", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.8, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "hnsw_recall_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: HNSW latency comparison
# ---------------------------------------------------------------------------
def plot_hnsw_latency(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, cfg in DATASETS.items():
        rows = data[name]
        vals, ns = [], []
        for n in sizes:
            r = _get_method_row(rows, n, "hnsw")
            if r:
                vals.append(r["mean_latency"] * 1000)
                ns.append(n)
        if vals:
            ax.plot(ns, vals, marker="D", linewidth=2, markersize=8,
                    color=cfg["color"], label=cfg["label"])

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("HNSW Mean Latency (ms)", fontsize=11)
    ax.set_title("HNSW Query Latency by Dataset", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "hnsw_latency_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Best LSH recall comparison
# ---------------------------------------------------------------------------
def plot_best_lsh_recall(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, cfg in DATASETS.items():
        rows = data[name]
        vals, ns = [], []
        for n in sizes:
            r = _best_lsh(rows, n)
            if r:
                vals.append(r["recall"])
                ns.append(n)
        if vals:
            ax.plot(ns, vals, marker="o", linewidth=2, markersize=8,
                    color=cfg["color"], label=cfg["label"])

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel(f"Best LSH Recall@{K}", fontsize=11)
    ax.set_title(f"Best LSH Config Recall@{K} by Dataset", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.8, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "best_lsh_recall_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Best LSH latency comparison
# ---------------------------------------------------------------------------
def plot_best_lsh_latency(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, cfg in DATASETS.items():
        rows = data[name]
        vals, ns = [], []
        for n in sizes:
            r = _best_lsh(rows, n)
            if r:
                vals.append(r["mean_latency"] * 1000)
                ns.append(n)
        if vals:
            ax.plot(ns, vals, marker="o", linewidth=2, markersize=8,
                    color=cfg["color"], label=cfg["label"])

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Best LSH Mean Latency (ms)", fontsize=11)
    ax.set_title("Best LSH Config Query Latency by Dataset", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "best_lsh_latency_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Build time comparison (HNSW vs best LSH per dataset)
# ---------------------------------------------------------------------------
def plot_build_time(data, sizes):
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"sift": "-", "glove": "--"}
    for name, cfg in DATASETS.items():
        rows = data[name]
        for method_name, get_fn, marker in [
            ("HNSW", lambda rows, n: _get_method_row(rows, n, "hnsw"), "D"),
            ("Best LSH", lambda rows, n: _best_lsh(rows, n), "o"),
        ]:
            vals, ns = [], []
            for n in sizes:
                r = get_fn(rows, n)
                if r:
                    vals.append(r["build_time"])
                    ns.append(n)
            if vals:
                ax.plot(ns, vals, marker=marker, linewidth=2, markersize=7,
                        color=cfg["color"], linestyle=styles[name],
                        label=f"{cfg['label']} {method_name}")

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Build Time (s)", fontsize=11)
    ax.set_title("Index Build Time by Dataset", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "build_time_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: Recall vs Latency Pareto (side-by-side per common N)
# ---------------------------------------------------------------------------
def plot_recall_vs_latency_sidebyside(data, sizes):
    for n in sizes:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 6), sharey=True)
        if len(DATASETS) == 1:
            axes = [axes]

        for ax, (name, cfg) in zip(axes, DATASETS.items()):
            rows = [r for r in data[name] if r["n"] == n]

            fs = [r for r in rows if r["method"] == "full_scan"]
            if fs:
                ax.scatter([r["mean_latency"] * 1000 for r in fs], [r["recall"] for r in fs],
                           marker="s", s=160, color="black", label="Full Scan", zorder=10,
                           edgecolors="black", linewidths=1.5)

            hnsw = [r for r in rows if r["method"] == "hnsw"]
            if hnsw:
                ax.scatter([r["mean_latency"] * 1000 for r in hnsw], [r["recall"] for r in hnsw],
                           marker="D", s=160, color="#e41a1c", label="HNSW", zorder=10,
                           edgecolors="black", linewidths=1.5)

            lsh = [r for r in rows if r["method"].startswith("lsh_")]
            for r in lsh:
                ax.scatter(r["mean_latency"] * 1000, r["recall"], marker="o", s=50,
                           color=cfg["color"], alpha=0.6, edgecolors="black", linewidths=0.3)

            ax.set_xlabel("Mean Query Latency (ms)", fontsize=11)
            ax.set_title(f"{cfg['label']}  (N={n:,})", fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-0.05, 1.08)

        axes[0].set_ylabel(f"Recall@{K}", fontsize=11)
        fig.suptitle(f"Recall vs Latency Comparison  (N={n:,}, K={K})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(COMPARISON_DIR, f"recall_vs_latency_n{n}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 7: Summary bar chart at largest common N
# ---------------------------------------------------------------------------
def plot_summary_comparison(data, sizes):
    max_n = max(sizes)
    methods_per_dataset = {}

    for name, cfg in DATASETS.items():
        rows = data[name]
        fs = _get_method_row(rows, max_n, "full_scan")
        hnsw = _get_method_row(rows, max_n, "hnsw")
        best = _best_lsh(rows, max_n)
        methods_per_dataset[name] = {"full_scan": fs, "hnsw": hnsw, "best_lsh": best}

    method_labels = ["Full Scan", "HNSW", "Best LSH"]
    method_keys = ["full_scan", "hnsw", "best_lsh"]
    metrics = [
        (f"Recall@{K}", "recall", 1, "{:.2f}"),
        ("Mean Latency (ms)", "mean_latency", 1000, "{:.1f}"),
        ("Build Time (s)", "build_time", 1, "{:.2f}"),
        ("Throughput (q/s)", "throughput", 1, "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(method_labels))
    width = 0.35

    for ax, (title, key, mult, fmt) in zip(axes, metrics):
        for i, (name, cfg) in enumerate(DATASETS.items()):
            vals = []
            for mk in method_keys:
                r = methods_per_dataset[name].get(mk)
                vals.append(r[key] * mult if r else 0)
            offset = -width / 2 + i * width
            bars = ax.bar(x + offset, vals, width, label=cfg["label"],
                          color=cfg["color"], edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            fmt.format(val), ha="center", va="bottom", fontsize=7,
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Dataset Comparison Summary  (N={max_n:,}, K={K})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(COMPARISON_DIR, "summary_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 8: LSH recall heatmap difference (GloVe - SIFT)
# ---------------------------------------------------------------------------
def plot_lsh_recall_diff_heatmap(data, sizes):
    tables_grid = [2, 4, 8, 16, 32]
    bits_grid = [8, 12, 16, 24, 32]

    for n in sizes:
        grids = {}
        for name in DATASETS:
            grid = np.full((len(tables_grid), len(bits_grid)), np.nan)
            for r in data[name]:
                if r["n"] == n and r["lsh_tables"] is not None:
                    ti = tables_grid.index(r["lsh_tables"]) if r["lsh_tables"] in tables_grid else -1
                    bi = bits_grid.index(r["lsh_bits"]) if r["lsh_bits"] in bits_grid else -1
                    if ti >= 0 and bi >= 0:
                        grid[ti, bi] = r["recall"]
            grids[name] = grid

        if len(grids) < 2:
            continue

        names = list(DATASETS.keys())
        diff = grids[names[1]] - grids[names[0]]

        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.01)
        im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(bits_grid)))
        ax.set_xticklabels(bits_grid, fontsize=10)
        ax.set_yticks(range(len(tables_grid)))
        ax.set_yticklabels(tables_grid, fontsize=10)
        ax.set_xlabel("lsh_bits", fontsize=11)
        ax.set_ylabel("lsh_tables", fontsize=11)
        ax.set_title(f"LSH Recall Difference ({DATASETS[names[1]]['label']} − "
                     f"{DATASETS[names[0]]['label']})  N={n:,}", fontsize=12)
        for i in range(len(tables_grid)):
            for j in range(len(bits_grid)):
                val = diff[i, j]
                if not np.isnan(val):
                    color = "black" if abs(val) < vmax * 0.5 else "white"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, label="Recall Difference", shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(COMPARISON_DIR, f"lsh_recall_diff_n{n}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    data = {}
    for name, cfg in DATASETS.items():
        if not os.path.exists(cfg["csv"]):
            print(f"WARNING: {cfg['csv']} not found. Run: python benchmark.py --dataset {name}")
            continue
        data[name] = load_csv(cfg["csv"])
        print(f"Loaded {len(data[name])} rows for {cfg['label']}")

    if len(data) < 2:
        print("Need results from both datasets to compare. Run benchmarks first.")
        return

    sizes = common_sizes(data)
    if not sizes:
        print("No common dataset sizes found between benchmarks.")
        return

    print(f"Common sizes: {sizes}")
    print("Generating comparison charts...")

    plot_hnsw_recall(data, sizes)
    plot_hnsw_latency(data, sizes)
    plot_best_lsh_recall(data, sizes)
    plot_best_lsh_latency(data, sizes)
    plot_build_time(data, sizes)
    plot_recall_vs_latency_sidebyside(data, sizes)
    plot_summary_comparison(data, sizes)
    plot_lsh_recall_diff_heatmap(data, sizes)

    print(f"Comparison charts saved to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
