#!/usr/bin/env python3
"""
Benchmark: Full Scan vs HNSW vs LSH in DuckDB
Evaluates query latency, recall@K, index build time, and scalability.

Usage:
    python benchmark.py --dataset sift
    python benchmark.py --dataset glove
"""

import argparse
import csv
import os
import struct
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSION_PATH = os.path.join(
    SCRIPT_DIR, "..", "build", "release", "extension", "vss", "vss.duckdb_extension"
)

K = 10
NUM_QUERIES = 200
LSH_TABLES_GRID = [2, 4, 8, 16, 32]
LSH_BITS_GRID = [8, 12, 16, 24, 32]

DATASET_CONFIG = {
    "sift": {
        "data_dir": os.path.join(SCRIPT_DIR, "data", "sift"),
        "base": "sift_base.fvecs",
        "query": "sift_query.fvecs",
        "groundtruth": "sift_groundtruth.ivecs",
        "sizes": [10_000, 50_000, 100_000, 500_000, 1_000_000],
        "label": "SIFT-128d",
    },
    "glove": {
        "data_dir": os.path.join(SCRIPT_DIR, "data", "glove"),
        "base": "glove_base.fvecs",
        "query": "glove_query.fvecs",
        "groundtruth": "glove_groundtruth.ivecs",
        "sizes": [10_000, 50_000, 100_000, 500_000, 1_000_000],
        "label": "GloVe-200d",
    },
}

# ---------------------------------------------------------------------------
# fvecs / ivecs parsing
# ---------------------------------------------------------------------------

def read_fvecs(path):
    with open(path, "rb") as f:
        data = f.read()
    offset, vectors = 0, []
    while offset < len(data):
        (dim,) = struct.unpack_from("<i", data, offset)
        offset += 4
        vec = struct.unpack_from(f"<{dim}f", data, offset)
        offset += dim * 4
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_ivecs(path):
    with open(path, "rb") as f:
        data = f.read()
    offset, vectors = 0, []
    while offset < len(data):
        (dim,) = struct.unpack_from("<i", data, offset)
        offset += 4
        vec = struct.unpack_from(f"<{dim}i", data, offset)
        offset += dim * 4
        vectors.append(vec)
    return np.array(vectors, dtype=np.int32)

# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------

def get_connection():
    conn = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    conn.execute(f"LOAD '{EXTENSION_PATH}'")
    return conn


def load_table(conn, base_vectors, n, dim):
    conn.execute("DROP TABLE IF EXISTS vectors")
    conn.execute(f"CREATE TABLE vectors (id INTEGER, embedding FLOAT[{dim}])")
    batch = 10_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        rows = []
        for i in range(start, end):
            vec_str = "[" + ",".join(str(float(x)) for x in base_vectors[i]) + "]"
            rows.append(f"({i}, {vec_str}::FLOAT[{dim}])")
        conn.execute(f"INSERT INTO vectors VALUES {','.join(rows)}")


def query_vec_sql(vec, dim):
    return "[" + ",".join(str(float(x)) for x in vec) + f"]::FLOAT[{dim}]"


def run_queries(conn, queries, k, dim):
    all_results, latencies = [], []
    for q in queries:
        q_sql = query_vec_sql(q, dim)
        sql = f"SELECT id FROM vectors ORDER BY array_distance(embedding, {q_sql}) LIMIT {k}"
        t0 = time.perf_counter()
        res = conn.execute(sql).fetchall()
        t1 = time.perf_counter()
        all_results.append([r[0] for r in res])
        latencies.append(t1 - t0)
    return all_results, latencies

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_recall(results, ground_truth, k):
    recalls = []
    for res, gt in zip(results, ground_truth):
        gt_set = set(int(x) for x in gt[:k])
        res_set = set(res[:k])
        recalls.append(len(res_set & gt_set) / k)
    return np.mean(recalls)

# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_full_scan(conn, queries, k, dim):
    print("    Full scan...")
    results, latencies = run_queries(conn, queries, k, dim)
    return {
        "method": "full_scan",
        "build_time": 0.0,
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": len(queries) / sum(latencies),
        "recall": 1.0,
    }, results


def bench_hnsw(conn, queries, ground_truth, k, dim):
    print("    HNSW...")
    conn.execute("DROP INDEX IF EXISTS hnsw_idx")
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX hnsw_idx ON vectors USING HNSW (embedding)")
    build_time = time.perf_counter() - t0
    results, latencies = run_queries(conn, queries, k, dim)
    recall = compute_recall(results, ground_truth, k)
    conn.execute("DROP INDEX IF EXISTS hnsw_idx")
    return {
        "method": "hnsw",
        "build_time": build_time,
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": len(queries) / sum(latencies),
        "recall": recall,
    }


def bench_lsh(conn, queries, ground_truth, k, dim, num_tables, num_bits):
    label = f"lsh_t{num_tables}_b{num_bits}"
    print(f"    LSH (tables={num_tables}, bits={num_bits})...")
    conn.execute("DROP INDEX IF EXISTS lsh_idx")
    t0 = time.perf_counter()
    conn.execute(
        f"CREATE INDEX lsh_idx ON vectors USING LSH (embedding) "
        f"WITH (lsh_tables={num_tables}, lsh_bits={num_bits})"
    )
    build_time = time.perf_counter() - t0
    results, latencies = run_queries(conn, queries, k, dim)
    recall = compute_recall(results, ground_truth, k)
    conn.execute("DROP INDEX IF EXISTS lsh_idx")
    return {
        "method": label,
        "build_time": build_time,
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": len(queries) / sum(latencies),
        "recall": recall,
        "lsh_tables": num_tables,
        "lsh_bits": num_bits,
    }

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_TABLE_COLORS = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c", 16: "#d62728", 32: "#9467bd"}
_BITS_SIZES = {8: 30, 12: 55, 16: 80, 24: 110, 32: 145}


def _get_lsh_style(r):
    return _TABLE_COLORS.get(r.get("lsh_tables"), "gray"), _BITS_SIZES.get(r.get("lsh_bits"), 60)


def _pick_representative_lsh(all_rows):
    lsh = [r for r in all_rows if r["method"].startswith("lsh_")]
    if not lsh:
        return []
    by_recall = sorted(lsh, key=lambda r: r["recall"])
    best, worst, mid = by_recall[-1], by_recall[0], by_recall[len(by_recall) // 2]
    seen, reps = set(), []
    for r in [best, mid, worst]:
        if r["method"] not in seen:
            seen.add(r["method"])
            reps.append(r)
    return reps

# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_recall_vs_latency(all_rows, n, plots_dir, dataset_label):
    fig, ax = plt.subplots(figsize=(9, 6))
    rows = [r for r in all_rows if r["n"] == n]

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
    tables_seen, bits_seen = set(), set()
    for r in lsh:
        c, s = _get_lsh_style(r)
        ax.scatter(r["mean_latency"] * 1000, r["recall"], marker="o", s=s, color=c,
                   alpha=0.85, zorder=5, edgecolors="black", linewidths=0.5,
                   label=f"tables={r['lsh_tables']}" if r["lsh_tables"] not in tables_seen else None)
        tables_seen.add(r["lsh_tables"])
        bits_seen.add(r["lsh_bits"])

    if lsh:
        best = max(lsh, key=lambda r: (r["recall"], -r["mean_latency"]))
        ax.annotate(f"  t={best['lsh_tables']}, b={best['lsh_bits']}",
                    (best["mean_latency"] * 1000, best["recall"]), fontsize=8, fontweight="bold")

    for b in sorted(bits_seen):
        ax.scatter([], [], marker="o", s=_BITS_SIZES.get(b, 60), color="gray",
                   edgecolors="black", linewidths=0.5, label=f"bits={b}")

    ax.set_xlabel("Mean Query Latency (ms)", fontsize=11)
    ax.set_ylabel(f"Recall@{K}", fontsize=11)
    ax.set_title(f"{dataset_label}: Recall vs Query Latency  (N={n:,}, K={K})", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"recall_vs_latency_n{n}.png"), dpi=150)
    plt.close(fig)


def plot_build_time(all_rows, plots_dir, dataset_label):
    fig, ax = plt.subplots(figsize=(9, 6))
    hnsw = sorted([r for r in all_rows if r["method"] == "hnsw"], key=lambda r: r["n"])
    if hnsw:
        ax.plot([r["n"] for r in hnsw], [r["build_time"] for r in hnsw],
                marker="D", color="#e41a1c", linewidth=2, markersize=8, label="HNSW")

    sizes = sorted(set(r["n"] for r in all_rows))
    lsh_methods = set()
    for n in sizes:
        for r in _pick_representative_lsh([r for r in all_rows if r["n"] == n]):
            lsh_methods.add(r["method"])

    for method in sorted(lsh_methods):
        rows = sorted([r for r in all_rows if r["method"] == method], key=lambda r: r["n"])
        if rows:
            t, b = rows[0].get("lsh_tables", "?"), rows[0].get("lsh_bits", "?")
            ax.plot([r["n"] for r in rows], [r["build_time"] for r in rows],
                    marker="o", color=_TABLE_COLORS.get(t, "gray"), linewidth=1.5,
                    markersize=6, linestyle="--", label=f"LSH t={t} b={b}")

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Index Build Time (s)", fontsize=11)
    ax.set_title(f"{dataset_label}: Index Build Time vs Dataset Size", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "build_time.png"), dpi=150)
    plt.close(fig)


def plot_scalability(all_rows, plots_dir, dataset_label):
    fig, ax = plt.subplots(figsize=(9, 6))
    fs = sorted([r for r in all_rows if r["method"] == "full_scan"], key=lambda r: r["n"])
    if fs:
        ax.plot([r["n"] for r in fs], [r["mean_latency"] * 1000 for r in fs],
                marker="s", color="black", linewidth=2, markersize=8, label="Full Scan")

    hnsw = sorted([r for r in all_rows if r["method"] == "hnsw"], key=lambda r: r["n"])
    if hnsw:
        ax.plot([r["n"] for r in hnsw], [r["mean_latency"] * 1000 for r in hnsw],
                marker="D", color="#e41a1c", linewidth=2, markersize=8, label="HNSW")

    sizes = sorted(set(r["n"] for r in all_rows))
    lsh_methods = set()
    for n in sizes:
        for r in _pick_representative_lsh([r for r in all_rows if r["n"] == n]):
            lsh_methods.add(r["method"])

    for method in sorted(lsh_methods):
        rows = sorted([r for r in all_rows if r["method"] == method], key=lambda r: r["n"])
        if rows:
            t, b = rows[0].get("lsh_tables", "?"), rows[0].get("lsh_bits", "?")
            ax.plot([r["n"] for r in rows], [r["mean_latency"] * 1000 for r in rows],
                    marker="o", color=_TABLE_COLORS.get(t, "gray"), linewidth=1.5,
                    markersize=6, linestyle="--", label=f"LSH t={t} b={b}")

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Mean Query Latency (ms)", fontsize=11)
    ax.set_title(f"{dataset_label}: Query Latency vs Dataset Size  (K={K})", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "scalability.png"), dpi=150)
    plt.close(fig)


def plot_lsh_param_sensitivity(all_rows, n, plots_dir, dataset_label):
    lsh_rows = [r for r in all_rows if r["method"].startswith("lsh_") and r["n"] == n]
    if not lsh_rows:
        return

    tables_vals = sorted(set(r["lsh_tables"] for r in lsh_rows))
    bits_vals = sorted(set(r["lsh_bits"] for r in lsh_rows))

    recall_grid = np.full((len(tables_vals), len(bits_vals)), np.nan)
    latency_grid = np.full((len(tables_vals), len(bits_vals)), np.nan)
    for r in lsh_rows:
        ti, bi = tables_vals.index(r["lsh_tables"]), bits_vals.index(r["lsh_bits"])
        recall_grid[ti, bi] = r["recall"]
        latency_grid[ti, bi] = r["mean_latency"] * 1000

    for grid, cmap, vlabel, fname, vmin, vmax in [
        (recall_grid, "RdYlGn", f"Recall@{K}", f"lsh_recall_heatmap_n{n}.png", 0, 1),
        (latency_grid, "YlOrRd", "Latency (ms)", f"lsh_latency_heatmap_n{n}.png", None, None),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(bits_vals)))
        ax.set_xticklabels(bits_vals, fontsize=10)
        ax.set_yticks(range(len(tables_vals)))
        ax.set_yticklabels(tables_vals, fontsize=10)
        ax.set_xlabel("lsh_bits (hash length)", fontsize=11)
        ax.set_ylabel("lsh_tables (number of hash tables)", fontsize=11)
        title_metric = "Recall" if "recall" in fname else "Mean Query Latency (ms)"
        ax.set_title(f"{dataset_label}: LSH {title_metric}  (N={n:,})", fontsize=13)
        for i in range(len(tables_vals)):
            for j in range(len(bits_vals)):
                val = grid[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.2f}" if "recall" in fname else f"{val:.1f}"
                    color = "white" if ("recall" in fname and val < 0.5) else "black"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=10,
                            fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, label=vlabel, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)


def plot_throughput(all_rows, n, plots_dir, dataset_label):
    rows = [r for r in all_rows if r["n"] == n]
    if not rows:
        return

    fs = next((r for r in rows if r["method"] == "full_scan"), None)
    hnsw = next((r for r in rows if r["method"] == "hnsw"), None)
    lsh_all = [r for r in rows if r["method"].startswith("lsh_")]
    best_lsh = max(lsh_all, key=lambda r: r["recall"]) if lsh_all else None

    entries = []
    if fs:
        entries.append(("Full Scan", fs["throughput"], "black"))
    if hnsw:
        entries.append(("HNSW", hnsw["throughput"], "#e41a1c"))
    if best_lsh:
        entries.append((f"LSH t={best_lsh['lsh_tables']} b={best_lsh['lsh_bits']}",
                        best_lsh["throughput"], "#2ca02c"))
    if not entries:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (queries/sec)", fontsize=11)
    ax.set_title(f"{dataset_label}: Query Throughput  (N={n:,}, K={K})", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"throughput_n{n}.png"), dpi=150)
    plt.close(fig)


def plot_summary(all_rows, plots_dir, dataset_label):
    max_n = max(r["n"] for r in all_rows)
    rows = [r for r in all_rows if r["n"] == max_n]

    fs = next((r for r in rows if r["method"] == "full_scan"), None)
    hnsw = next((r for r in rows if r["method"] == "hnsw"), None)
    lsh_all = [r for r in rows if r["method"].startswith("lsh_")]
    best_lsh = max(lsh_all, key=lambda r: r["recall"]) if lsh_all else None
    worst_lsh = min(lsh_all, key=lambda r: r["recall"]) if lsh_all else None

    methods = []
    if fs:
        methods.append(("Full Scan", fs, "black"))
    if hnsw:
        methods.append(("HNSW", hnsw, "#e41a1c"))
    if best_lsh:
        methods.append((f"LSH best\nt={best_lsh['lsh_tables']} b={best_lsh['lsh_bits']}",
                        best_lsh, "#2ca02c"))
    if worst_lsh and worst_lsh["method"] != (best_lsh or {}).get("method"):
        methods.append((f"LSH worst\nt={worst_lsh['lsh_tables']} b={worst_lsh['lsh_bits']}",
                        worst_lsh, "#ff7f0e"))
    if len(methods) < 2:
        return

    metrics = [
        (f"Recall@{K}", "recall", "{:.2f}"),
        ("Mean Latency (ms)", "mean_latency", "{:.1f}"),
        ("Build Time (s)", "build_time", "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (title, key, fmt) in zip(axes, metrics):
        names = [m[0] for m in methods]
        vals = [m[1][key] * (1000 if key == "mean_latency" else 1) for m in methods]
        colors = [m[2] for m in methods]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(f"{dataset_label}: Summary  (N={max_n:,}, K={K})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Full Scan vs HNSW vs LSH")
    parser.add_argument("--dataset", required=True, choices=DATASET_CONFIG.keys(),
                        help="Dataset to benchmark (sift or glove)")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    results_dir = os.path.join(SCRIPT_DIR, args.dataset, "results")
    plots_dir = os.path.join(SCRIPT_DIR, args.dataset, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading {cfg['label']} data...")
    base_vectors = read_fvecs(os.path.join(cfg["data_dir"], cfg["base"]))
    query_vectors = read_fvecs(os.path.join(cfg["data_dir"], cfg["query"]))
    ground_truth = read_ivecs(os.path.join(cfg["data_dir"], cfg["groundtruth"]))

    dim = base_vectors.shape[1]
    print(f"  Base: {base_vectors.shape}, Queries: {query_vectors.shape}, "
          f"GT: {ground_truth.shape}, dim={dim}")

    queries = query_vectors[:NUM_QUERIES]
    all_rows = []
    csv_fields = [
        "n", "method", "build_time", "mean_latency", "median_latency",
        "p99_latency", "throughput", "recall", "lsh_tables", "lsh_bits",
    ]

    for n in cfg["sizes"]:
        if n > base_vectors.shape[0]:
            print(f"Skipping N={n}, only {base_vectors.shape[0]} base vectors available")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset size: N={n:,}")
        print(f"{'='*60}")

        conn = get_connection()
        print("  Loading table...")
        load_table(conn, base_vectors, n, dim)

        row, exact_results = bench_full_scan(conn, queries, K, dim)
        row["n"] = n
        row.setdefault("lsh_tables", "")
        row.setdefault("lsh_bits", "")
        all_rows.append(row)
        print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms")

        row = bench_hnsw(conn, queries, exact_results, K, dim)
        row["n"] = n
        row.setdefault("lsh_tables", "")
        row.setdefault("lsh_bits", "")
        all_rows.append(row)
        print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms  "
              f"build={row['build_time']:.2f}s")

        for num_tables in LSH_TABLES_GRID:
            for num_bits in LSH_BITS_GRID:
                row = bench_lsh(conn, queries, exact_results, K, dim, num_tables, num_bits)
                row["n"] = n
                all_rows.append(row)
                print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms  "
                      f"build={row['build_time']:.2f}s")

        conn.close()

    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in csv_fields})
    print(f"\nResults saved to {csv_path}")

    print("Generating plots...")
    for n in cfg["sizes"]:
        if any(r["n"] == n for r in all_rows):
            plot_recall_vs_latency(all_rows, n, plots_dir, cfg["label"])
            plot_lsh_param_sensitivity(all_rows, n, plots_dir, cfg["label"])
            plot_throughput(all_rows, n, plots_dir, cfg["label"])
    plot_build_time(all_rows, plots_dir, cfg["label"])
    plot_scalability(all_rows, plots_dir, cfg["label"])
    plot_summary(all_rows, plots_dir, cfg["label"])
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
