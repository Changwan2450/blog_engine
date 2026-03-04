"""backtest.py – Evaluate historical ARM/LENS/term performance.

Loads past runs from data/runs.jsonl and metrics from data/metrics.csv,
joins them by output_file, and computes average reward per:
  - arm
  - lens
  - arm|lens
  - term

Outputs a console summary and a markdown report to out/backtest_report.md.

Usage:
    python3 src/backtest.py
    python3 src/backtest.py --data-dir data --out-dir out
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_runs(runs_path: Path) -> list[dict]:
    """Load all non-metric_update records from runs.jsonl."""
    if not runs_path.exists():
        return []
    records = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("slot") == "metric_update":
            continue
        records.append(rec)
    return records


def _load_metrics(csv_path: Path) -> dict[str, list[dict]]:
    """Load metrics.csv and index by output_file."""
    if not csv_path.exists():
        return {}
    index: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("output_file", "").strip()
            if key:
                index[key].append(row)
    return dict(index)


def _rel_path(raw: str, root: Path) -> str:
    """Normalise to project-root-relative path."""
    try:
        p = Path(raw).resolve()
        return str(p.relative_to(root))
    except (ValueError, OSError):
        return raw


# ---------------------------------------------------------------------------
# Join + aggregate
# ---------------------------------------------------------------------------

def _join_data(
    runs: list[dict],
    metrics: dict[str, list[dict]],
    project_root: Path,
) -> list[dict]:
    """Join runs with their metrics by output_file."""
    joined = []
    for run in runs:
        out_file = run.get("output_file", "")
        out_rel = run.get("output_file_rel", "") or _rel_path(out_file, project_root)

        # Find matching metrics
        matched = metrics.get(out_file) or metrics.get(out_rel) or []
        if not matched:
            # Try relative version of each metric key
            for mk, mv in metrics.items():
                if _rel_path(mk, project_root) == out_rel:
                    matched = mv
                    break

        for m in matched:
            try:
                reward = float(m.get("reward", 0))
            except (ValueError, TypeError):
                reward = 0.0
            joined.append({
                "arm": run.get("arm", m.get("arm", "")),
                "lens": run.get("lens", m.get("lens", "")),
                "term1": m.get("term1", ""),
                "term2": m.get("term2", ""),
                "reward": reward,
                "output_file": out_rel,
                "timestamp": run.get("timestamp", ""),
            })

    return joined


def _aggregate(joined: list[dict]) -> dict[str, dict[str, dict]]:
    """Compute average reward per dimension.

    Returns: {dimension_name: {key: {"total": float, "count": int, "avg": float}}}
    """
    dims: dict[str, dict[str, dict]] = {
        "arm": defaultdict(lambda: {"total": 0.0, "count": 0}),
        "lens": defaultdict(lambda: {"total": 0.0, "count": 0}),
        "arm|lens": defaultdict(lambda: {"total": 0.0, "count": 0}),
        "term": defaultdict(lambda: {"total": 0.0, "count": 0}),
    }

    for rec in joined:
        arm = rec["arm"]
        lens = rec["lens"]
        reward = rec["reward"]

        if arm:
            dims["arm"][arm]["total"] += reward
            dims["arm"][arm]["count"] += 1

        if lens:
            dims["lens"][lens]["total"] += reward
            dims["lens"][lens]["count"] += 1

        if arm and lens:
            key = f"{arm}|{lens}"
            dims["arm|lens"][key]["total"] += reward
            dims["arm|lens"][key]["count"] += 1

        for t in [rec.get("term1", ""), rec.get("term2", "")]:
            t = t.strip()
            if t:
                dims["term"][t]["total"] += reward
                dims["term"][t]["count"] += 1

    # Compute averages
    for dim_data in dims.values():
        for entry in dim_data.values():
            entry["avg"] = round(entry["total"] / max(1, entry["count"]), 4)

    return dims


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _sorted_entries(dim_data: dict, top_n: int = 10) -> list[tuple[str, dict]]:
    """Sort dimension entries by average reward descending, take top N."""
    return sorted(dim_data.items(), key=lambda x: x[1]["avg"], reverse=True)[:top_n]


def _print_console(dims: dict[str, dict[str, dict]], total_records: int) -> None:
    """Print a console summary."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST REPORT  ({total_records} metric records)")
    print(f"{'='*60}\n")

    labels = {
        "arm": "Top Arms",
        "lens": "Top Lenses",
        "arm|lens": "Top Arm|Lens Combos",
        "term": "Top Terms",
    }

    for dim_name, label in labels.items():
        entries = _sorted_entries(dims[dim_name])
        if not entries:
            print(f"  {label}: (no data)\n")
            continue
        print(f"  {label}:")
        for key, data in entries:
            print(f"    {key:20s}  avg={data['avg']:+.4f}  n={data['count']}")
        print()


def _write_report(
    dims: dict[str, dict[str, dict]],
    total_records: int,
    out_path: Path,
) -> Path:
    """Write a markdown report to out/backtest_report.md."""
    lines = [
        f"# Backtest Report",
        "",
        f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Total metric records: {total_records}",
        "",
    ]

    sections = {
        "arm": "Arms",
        "lens": "Lenses",
        "arm|lens": "Arm|Lens Combos",
        "term": "Terms",
    }

    for dim_name, title in sections.items():
        entries = _sorted_entries(dims[dim_name], top_n=15)
        lines.append(f"## Top {title}")
        lines.append("")
        if not entries:
            lines.append("_(no data)_")
            lines.append("")
            continue
        lines.append("| Rank | Key | Avg Reward | Count |")
        lines.append("|------|-----|-----------|-------|")
        for i, (key, data) in enumerate(entries, 1):
            lines.append(f"| {i} | `{key}` | {data['avg']:+.4f} | {data['count']} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Interpretation",
        "",
        "- **Avg Reward**: mean reward across all metric updates for that key.",
        "- **Count**: number of metric update events contributing to the average.",
        "- Higher avg reward → historically better engagement.",
        "- Low count values may not be statistically significant.",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest blog engine arm/lens/term performance")
    parser.add_argument("--data-dir", default=None, help="Path to data/ directory")
    parser.add_argument("--out-dir", default=None, help="Path to out/ directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "out"

    runs_path = data_dir / "runs.jsonl"
    csv_path = data_dir / "metrics.csv"

    # Load
    runs = _load_runs(runs_path)
    metrics = _load_metrics(csv_path)

    if not runs:
        print("[backtest] No run records found in runs.jsonl", file=sys.stderr)
    if not metrics:
        print("[backtest] No metric records found in metrics.csv", file=sys.stderr)

    # Join
    joined = _join_data(runs, metrics, project_root)

    if not joined:
        print("[backtest] No joined records. Run the pipeline and record metrics first.",
              file=sys.stderr)
        print("[backtest] Example: python3 src/learn.py --output_file out/..._final.md "
              "--views 100 --likes 5 --comments 2", file=sys.stderr)
        sys.exit(0)

    # Aggregate
    dims = _aggregate(joined)

    # Output
    _print_console(dims, len(joined))

    report_path = _write_report(dims, len(joined), out_dir / "backtest_report.md")
    print(f"📊  Report saved: {report_path}")


if __name__ == "__main__":
    main()
