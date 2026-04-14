#!/usr/bin/env python3
"""
Benchmark: Full Scan vs HNSW vs LSH in DuckDB
Evaluates query latency, recall@K, index build time, and scalability.
"""

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
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
EXTENSION_PATH = os.path.join(
    SCRIPT_DIR, "..", "build", "release", "extension", "vss", "vss.duckdb_extension"
)

K = 10
NUM_QUERIES = 200
DATASET_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
LSH_TABLES_GRID = [2, 4, 8, 16, 32]
LSH_BITS_GRID = [8, 12, 16, 24, 32]

# ---------------------------------------------------------------------------
# fvecs / ivecs parsing
# ---------------------------------------------------------------------------

def read_fvecs(path):
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vectors = []
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
    offset = 0
    vectors = []
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


def load_table(conn, base_vectors, n):
    """Create table and insert the first n base vectors."""
    conn.execute("DROP TABLE IF EXISTS sift")
    conn.execute("CREATE TABLE sift (id INTEGER, embedding FLOAT[128])")
    batch = 10_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        rows = []
        for i in range(start, end):
            vec_str = "[" + ",".join(str(float(x)) for x in base_vectors[i]) + "]"
            rows.append(f"({i}, {vec_str}::FLOAT[128])")
        conn.execute(f"INSERT INTO sift VALUES {','.join(rows)}")


def query_vec_sql(vec):
    return "[" + ",".join(str(float(x)) for x in vec) + "]::FLOAT[128]"


def run_queries(conn, queries, k):
    """Run top-K queries, return (list of result id lists, list of latencies)."""
    all_results = []
    latencies = []
    for q in queries:
        q_sql = query_vec_sql(q)
        sql = f"SELECT id FROM sift ORDER BY array_distance(embedding, {q_sql}) LIMIT {k}"
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
    """Average recall@K across queries."""
    recalls = []
    for res, gt in zip(results, ground_truth):
        gt_set = set(gt[:k])
        res_set = set(res[:k])
        recalls.append(len(res_set & gt_set) / k)
    return np.mean(recalls)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_full_scan(conn, queries, k):
    """Run full scan queries. Returns metrics dict AND the results (used as ground truth)."""
    print("    Full scan...")
    results, latencies = run_queries(conn, queries, k)
    return {
        "method": "full_scan",
        "build_time": 0.0,
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "throughput": len(queries) / sum(latencies),
        "recall": 1.0,  # full scan is exact by definition
    }, results


def bench_hnsw(conn, queries, ground_truth, k):
    print("    HNSW...")
    conn.execute("DROP INDEX IF EXISTS hnsw_idx")
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX hnsw_idx ON sift USING HNSW (embedding)")
    build_time = time.perf_counter() - t0

    results, latencies = run_queries(conn, queries, k)
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


def bench_lsh(conn, queries, ground_truth, k, num_tables, num_bits):
    label = f"lsh_t{num_tables}_b{num_bits}"
    print(f"    LSH (tables={num_tables}, bits={num_bits})...")
    conn.execute("DROP INDEX IF EXISTS lsh_idx")
    t0 = time.perf_counter()
    conn.execute(
        f"CREATE INDEX lsh_idx ON sift USING LSH (embedding) "
        f"WITH (lsh_tables={num_tables}, lsh_bits={num_bits})"
    )
    build_time = time.perf_counter() - t0

    results, latencies = run_queries(conn, queries, k)
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

# Color map: lsh_tables value -> color
_TABLE_COLORS = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c", 16: "#d62728", 32: "#9467bd"}
# Size map: lsh_bits value -> marker size
_BITS_SIZES = {8: 30, 12: 55, 16: 80, 24: 110, 32: 145}


def _get_lsh_style(r):
    """Return (color, size) for an LSH result row."""
    c = _TABLE_COLORS.get(r.get("lsh_tables"), "gray")
    s = _BITS_SIZES.get(r.get("lsh_bits"), 60)
    return c, s


def _pick_representative_lsh(all_rows):
    """Pick 3 LSH configs: best recall, worst recall, and a middle one."""
    lsh = [r for r in all_rows if r["method"].startswith("lsh_")]
    if not lsh:
        return []
    by_recall = sorted(lsh, key=lambda r: r["recall"])
    best = by_recall[-1]
    worst = by_recall[0]
    mid = by_recall[len(by_recall) // 2]
    seen = set()
    reps = []
    for r in [best, mid, worst]:
        if r["method"] not in seen:
            seen.add(r["method"])
            reps.append(r)
    return reps


# ---------------------------------------------------------------------------
# Plot 1: Recall vs Latency (per dataset size)
# ---------------------------------------------------------------------------

def plot_recall_vs_latency(all_rows, n):
    fig, ax = plt.subplots(figsize=(9, 6))
    rows = [r for r in all_rows if r["n"] == n]

    # --- Full scan ---
    fs = [r for r in rows if r["method"] == "full_scan"]
    if fs:
        ax.scatter(
            [r["mean_latency"] * 1000 for r in fs],
            [r["recall"] for r in fs],
            marker="s", s=160, color="black", label="Full Scan", zorder=10,
            edgecolors="black", linewidths=1.5,
        )

    # --- HNSW ---
    hnsw = [r for r in rows if r["method"] == "hnsw"]
    if hnsw:
        ax.scatter(
            [r["mean_latency"] * 1000 for r in hnsw],
            [r["recall"] for r in hnsw],
            marker="D", s=160, color="#e41a1c", label="HNSW", zorder=10,
            edgecolors="black", linewidths=1.5,
        )

    # --- LSH: color = tables, size = bits ---
    lsh = [r for r in rows if r["method"].startswith("lsh_")]
    tables_seen = set()
    bits_seen = set()
    for r in lsh:
        c, s = _get_lsh_style(r)
        t_label = f"tables={r['lsh_tables']}"
        ax.scatter(
            r["mean_latency"] * 1000, r["recall"],
            marker="o", s=s, color=c, alpha=0.85, zorder=5,
            edgecolors="black", linewidths=0.5,
            label=t_label if r["lsh_tables"] not in tables_seen else None,
        )
        tables_seen.add(r["lsh_tables"])
        bits_seen.add(r["lsh_bits"])

    # Annotate best LSH point
    if lsh:
        best = max(lsh, key=lambda r: (r["recall"], -r["mean_latency"]))
        ax.annotate(
            f"  t={best['lsh_tables']}, b={best['lsh_bits']}",
            (best["mean_latency"] * 1000, best["recall"]),
            fontsize=8, fontweight="bold",
        )

    # Size legend for bits
    if bits_seen:
        for b in sorted(bits_seen):
            ax.scatter([], [], marker="o", s=_BITS_SIZES.get(b, 60),
                       color="gray", edgecolors="black", linewidths=0.5,
                       label=f"bits={b}")

    ax.set_xlabel("Mean Query Latency (ms)", fontsize=11)
    ax.set_ylabel(f"Recall@{K}", fontsize=11)
    ax.set_title(f"Recall vs Query Latency  (N={n:,}, K={K})", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"recall_vs_latency_n{n}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Build time vs dataset size
# ---------------------------------------------------------------------------

def plot_build_time(all_rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    # HNSW
    hnsw = sorted(
        [r for r in all_rows if r["method"] == "hnsw"],
        key=lambda r: r["n"],
    )
    if hnsw:
        ax.plot([r["n"] for r in hnsw], [r["build_time"] for r in hnsw],
                marker="D", color="#e41a1c", linewidth=2, markersize=8, label="HNSW")

    # Representative LSH configs at each N
    sizes = sorted(set(r["n"] for r in all_rows))
    lsh_methods = set()
    for n in sizes:
        reps = _pick_representative_lsh([r for r in all_rows if r["n"] == n])
        for r in reps:
            lsh_methods.add(r["method"])

    for method in sorted(lsh_methods):
        rows = sorted(
            [r for r in all_rows if r["method"] == method],
            key=lambda r: r["n"],
        )
        if rows:
            t, b = rows[0].get("lsh_tables", "?"), rows[0].get("lsh_bits", "?")
            c = _TABLE_COLORS.get(t, "gray")
            ax.plot([r["n"] for r in rows], [r["build_time"] for r in rows],
                    marker="o", color=c, linewidth=1.5, markersize=6,
                    linestyle="--", label=f"LSH t={t} b={b}")

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Index Build Time (s)", fontsize=11)
    ax.set_title("Index Build Time vs Dataset Size", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "build_time.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Scalability — latency vs dataset size
# ---------------------------------------------------------------------------

def plot_scalability(all_rows):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Full scan
    fs = sorted([r for r in all_rows if r["method"] == "full_scan"], key=lambda r: r["n"])
    if fs:
        ax.plot([r["n"] for r in fs], [r["mean_latency"] * 1000 for r in fs],
                marker="s", color="black", linewidth=2, markersize=8, label="Full Scan")

    # HNSW
    hnsw = sorted([r for r in all_rows if r["method"] == "hnsw"], key=lambda r: r["n"])
    if hnsw:
        ax.plot([r["n"] for r in hnsw], [r["mean_latency"] * 1000 for r in hnsw],
                marker="D", color="#e41a1c", linewidth=2, markersize=8, label="HNSW")

    # Representative LSH configs
    sizes = sorted(set(r["n"] for r in all_rows))
    lsh_methods = set()
    for n in sizes:
        reps = _pick_representative_lsh([r for r in all_rows if r["n"] == n])
        for r in reps:
            lsh_methods.add(r["method"])

    for method in sorted(lsh_methods):
        rows = sorted([r for r in all_rows if r["method"] == method], key=lambda r: r["n"])
        if rows:
            t, b = rows[0].get("lsh_tables", "?"), rows[0].get("lsh_bits", "?")
            c = _TABLE_COLORS.get(t, "gray")
            ax.plot([r["n"] for r in rows], [r["mean_latency"] * 1000 for r in rows],
                    marker="o", color=c, linewidth=1.5, markersize=6,
                    linestyle="--", label=f"LSH t={t} b={b}")

    ax.set_xlabel("Dataset Size (N)", fontsize=11)
    ax.set_ylabel("Mean Query Latency (ms)", fontsize=11)
    ax.set_title(f"Query Latency vs Dataset Size  (K={K})", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "scalability.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: LSH parameter sensitivity heatmaps
# ---------------------------------------------------------------------------

def plot_lsh_param_sensitivity(all_rows, n):
    lsh_rows = [
        r for r in all_rows
        if r["method"].startswith("lsh_") and r["n"] == n
    ]
    if not lsh_rows:
        return

    tables_vals = sorted(set(r["lsh_tables"] for r in lsh_rows))
    bits_vals = sorted(set(r["lsh_bits"] for r in lsh_rows))

    recall_grid = np.full((len(tables_vals), len(bits_vals)), np.nan)
    latency_grid = np.full((len(tables_vals), len(bits_vals)), np.nan)
    for r in lsh_rows:
        ti = tables_vals.index(r["lsh_tables"])
        bi = bits_vals.index(r["lsh_bits"])
        recall_grid[ti, bi] = r["recall"]
        latency_grid[ti, bi] = r["mean_latency"] * 1000

    # --- Recall heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(recall_grid, aspect="auto", origin="lower", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(bits_vals)))
    ax.set_xticklabels(bits_vals, fontsize=10)
    ax.set_yticks(range(len(tables_vals)))
    ax.set_yticklabels(tables_vals, fontsize=10)
    ax.set_xlabel("lsh_bits (hash length)", fontsize=11)
    ax.set_ylabel("lsh_tables (number of hash tables)", fontsize=11)
    ax.set_title(f"LSH Recall@{K}  (N={n:,})", fontsize=13)
    for i in range(len(tables_vals)):
        for j in range(len(bits_vals)):
            val = recall_grid[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, label=f"Recall@{K}", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"lsh_recall_heatmap_n{n}.png"), dpi=150)
    plt.close(fig)

    # --- Latency heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(latency_grid, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_xticks(range(len(bits_vals)))
    ax.set_xticklabels(bits_vals, fontsize=10)
    ax.set_yticks(range(len(tables_vals)))
    ax.set_yticklabels(tables_vals, fontsize=10)
    ax.set_xlabel("lsh_bits (hash length)", fontsize=11)
    ax.set_ylabel("lsh_tables (number of hash tables)", fontsize=11)
    ax.set_title(f"LSH Mean Query Latency (ms)  (N={n:,})", fontsize=13)
    for i in range(len(tables_vals)):
        for j in range(len(bits_vals)):
            val = latency_grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Latency (ms)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"lsh_latency_heatmap_n{n}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Throughput comparison
# ---------------------------------------------------------------------------

def plot_throughput(all_rows, n):
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
        t, b = best_lsh["lsh_tables"], best_lsh["lsh_bits"]
        entries.append((f"LSH t={t} b={b}", best_lsh["throughput"], "#2ca02c"))

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
    ax.set_title(f"Query Throughput  (N={n:,}, K={K})", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"throughput_n{n}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Summary comparison bar chart (largest N)
# ---------------------------------------------------------------------------

def plot_summary(all_rows):
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
        t, b = best_lsh["lsh_tables"], best_lsh["lsh_bits"]
        methods.append((f"LSH best\nt={t} b={b}", best_lsh, "#2ca02c"))
    if worst_lsh and worst_lsh["method"] != (best_lsh or {}).get("method"):
        t, b = worst_lsh["lsh_tables"], worst_lsh["lsh_bits"]
        methods.append((f"LSH worst\nt={t} b={b}", worst_lsh, "#ff7f0e"))

    if len(methods) < 2:
        return

    metrics = [
        ("Recall@{}".format(K), "recall", "{:.2f}"),
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

    fig.suptitle(f"Summary Comparison  (N={max_n:,}, K={K})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SIFT data...")
    base_vectors = read_fvecs(os.path.join(DATA_DIR, "sift_base.fvecs"))
    query_vectors = read_fvecs(os.path.join(DATA_DIR, "sift_query.fvecs"))
    ground_truth = read_ivecs(os.path.join(DATA_DIR, "sift_groundtruth.ivecs"))

    print(f"  Base: {base_vectors.shape}, Queries: {query_vectors.shape}, GT: {ground_truth.shape}")

    queries = query_vectors[:NUM_QUERIES]

    all_rows = []
    csv_fields = [
        "n", "method", "build_time", "mean_latency", "median_latency",
        "p99_latency", "throughput", "recall", "lsh_tables", "lsh_bits",
    ]

    for n in DATASET_SIZES:
        if n > base_vectors.shape[0]:
            print(f"Skipping N={n}, only {base_vectors.shape[0]} base vectors available")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset size: N={n:,}")
        print(f"{'='*60}")

        conn = get_connection()
        print("  Loading table...")
        load_table(conn, base_vectors, n)

        # Full scan (exact baseline — results used as ground truth)
        row, exact_results = bench_full_scan(conn, queries, K)
        row["n"] = n
        row.setdefault("lsh_tables", "")
        row.setdefault("lsh_bits", "")
        all_rows.append(row)
        print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms")

        # HNSW
        row = bench_hnsw(conn, queries, exact_results, K)
        row["n"] = n
        row.setdefault("lsh_tables", "")
        row.setdefault("lsh_bits", "")
        all_rows.append(row)
        print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms  build={row['build_time']:.2f}s")

        # LSH parameter grid
        for num_tables in LSH_TABLES_GRID:
            for num_bits in LSH_BITS_GRID:
                row = bench_lsh(conn, queries, exact_results, K, num_tables, num_bits)
                row["n"] = n
                all_rows.append(row)
                print(f"    recall={row['recall']:.4f}  latency={row['mean_latency']*1000:.1f}ms  build={row['build_time']:.2f}s")

        conn.close()

    # Save results
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in csv_fields})
    print(f"\nResults saved to {csv_path}")

    # Generate plots
    print("Generating plots...")
    for n in DATASET_SIZES:
        if any(r["n"] == n for r in all_rows):
            plot_recall_vs_latency(all_rows, n)
            plot_lsh_param_sensitivity(all_rows, n)
            plot_throughput(all_rows, n)
    plot_build_time(all_rows)
    plot_scalability(all_rows)
    plot_summary(all_rows)
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
