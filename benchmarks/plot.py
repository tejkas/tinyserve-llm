"""
Figures and a summary table from a directory of runner results (the JSON files driver.py writes).

    python -m benchmarks.plot benchmarks/results/qwen --out benchmarks/results/qwen/plots

Repeats of one configuration are reduced to their median, per number and per wave. The dashed
"new tokens only" line is what a cache that keeps every history still has to prefill: each turn's new
user message (and the whole first prompt). It is read from the trace the runs replayed.

Figures, each a PNG and an SVG:
    prefill.*    total prefill tokens per engine at each pressure (the headline)
    by_turn.*    prefill tokens per turn, one panel per pressure
    ttft.*       median time to first token per turn, one panel per pressure
    throughput.* output tokens per second per engine at each pressure
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.multiturn import load_trace  # noqa: E402

# Series identity is fixed per engine, so a figure that drops one never repaints the rest.
SERIES = {
    "ours": ("tinyserve-llm", "#2a78d6"),
    "nano": ("nano-vllm", "#eb6834"),
    "off": ("tinyserve-llm, no prefix cache", "#8c8b86"),
}
ORDER = ["ours", "nano", "off"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def series_key(run: dict) -> str:
    if run["engine"] == "nano-vllm":
        return "nano"
    return "ours" if run["caching"] == "on" else "off"


def load_runs(results_dir: str) -> dict[tuple[str, float], list[dict]]:
    """(series, rho) -> runs."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        runs[series_key(run), float(run["rho_requested"])].append(run)
    return runs


def median(xs):
    return statistics.median(xs)


def per_wave(runs: list[dict], field: str) -> list[float]:
    """Median across repeats of one per-wave field, wave by wave."""
    columns = zip(*[[w[field] for w in r["waves"]] for r in runs], strict=True)
    return [median(c) for c in columns]


def new_tokens_by_wave(run: dict) -> list[int]:
    _, trace = load_trace(run["trace"])
    trace = trace[:run["conversations"]]
    waves = len(run["waves"])
    return [sum(len(c.turns[k].user_ids) for c in trace if len(c.turns) > k) for k in range(waves)]


def present(runs, rhos):
    return [k for k in ORDER if any((k, rho) in runs for rho in rhos)]


def bar_chart(runs, rhos, value, ylabel, title, fmt, path, floor=None):
    keys = present(runs, rhos)
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, key in enumerate(keys):
        label, color = SERIES[key]
        xs = [j + (i - (len(keys) - 1) / 2) * width for j, rho in enumerate(rhos) if (key, rho) in runs]
        ys = [value(runs[key, rho]) for rho in rhos if (key, rho) in runs]
        bars = ax.bar(xs, ys, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, [fmt(y) for y in ys], padding=3, fontsize=8.5, color=INK)
    if floor is not None:
        ax.axhline(floor, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="new tokens only")
    ax.set_xticks(range(len(rhos)), [f"ρ = {rho:g}" for rho in rhos])
    ax.set_xlabel("memory pressure: peak history ÷ KV capacity")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)
    header(fig, title, *ax.get_legend_handles_labels())
    save(fig, path)


def line_panels(runs, rhos, ys_for, ylabel, title, path, max_turn, floor_for=None, log=False):
    keys = present(runs, rhos)
    fig, axes = plt.subplots(1, len(rhos), figsize=(max(7.0, 3.4 * len(rhos)), 4.4), sharey=True, squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        for key in keys:
            if (key, rho) not in runs:
                continue
            ys = ys_for(runs[key, rho])[:max_turn]
            label, color = SERIES[key]
            ax.plot(range(1, len(ys) + 1), ys, color=color, linewidth=2, label=label)
        if floor_for is not None:
            ys = floor_for(rho)[:max_turn]
            ax.plot(range(1, len(ys) + 1), ys, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), label="new tokens only")
        ax.set_title(f"ρ = {rho:g}", loc="left", fontweight="normal")
        ax.set_xlabel("turn")
        if log:
            ax.set_yscale("log")
    axes[0][0].set_ylabel(ylabel)
    header(fig, title, *axes[0][0].get_legend_handles_labels())
    save(fig, path)


SUBTITLE = ""  # the workload, set once in main()


def header(fig, title, handles, labels):
    """Title, workload subtitle and a one-row legend, stacked above the plot area."""
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.925, SUBTITLE, ha="left", va="top", fontsize=9, color=MUTED)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.875), ncol=len(labels), fontsize=9,
               handlelength=1.6, columnspacing=1.4)


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def table(runs, rhos, floor) -> str:
    rows = ["| ρ | engine | prefill tokens | % of prompt | × new tokens | output tok/s | preemptions | runs |",
            "|---|---|---|---|---|---|---|---|"]
    for rho in rhos:
        for key in ORDER:
            if (key, rho) not in runs:
                continue
            rs = runs[key, rho]
            prefill = median([r["summary"]["prefill_tokens"] for r in rs])
            prompt = rs[0]["summary"]["prompt_tokens"]
            rows.append(f"| {rho:g} | {SERIES[key][0]} | {prefill:,.0f} | {prefill / prompt:.1%} | "
                        f"{prefill / floor:.2f} | {median([r['summary']['output_tokens_per_second'] for r in rs]):,.0f} | "
                        f"{median([r['summary']['preemptions'] for r in rs]):,.0f} | {len(rs)} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="directory of result JSON files")
    parser.add_argument("--out", default=None, help="default: <results>/plots")
    parser.add_argument("--max-turn", type=int, default=20, help="turns shown in the per-turn figures")
    args = parser.parse_args()
    out = args.out or os.path.join(args.results, "plots")
    os.makedirs(out, exist_ok=True)

    runs = load_runs(args.results)
    rhos = sorted({rho for _, rho in runs})
    first = next(iter(runs.values()))[0]
    for (key, rho), rs in runs.items():
        assert all(r["trace"] == first["trace"] and r["conversations"] == first["conversations"]
                   and r["max_waves"] == first["max_waves"] for r in rs), f"{key} rho {rho}: runs replay different workloads"
    new_by_wave = new_tokens_by_wave(first)
    floor = sum(new_by_wave)
    model = os.path.basename(first["model"].rstrip("/"))
    workload = f"{model}, {first['conversations']} ShareGPT conversations, {first['device']}"
    global SUBTITLE
    repeats = max(len(rs) for rs in runs.values())
    SUBTITLE = workload + (f"; median of {repeats} runs" if repeats > 1 else "; one run each")

    def total_prefill(rs):
        return median([r["summary"]["prefill_tokens"] for r in rs]) / 1e6

    bar_chart({k: v for k, v in runs.items() if k[0] != "off"}, rhos, total_prefill,
              "prefill tokens (millions)", "Tokens prefilled over the whole replay",
              lambda y: f"{y:.2f}M", os.path.join(out, "prefill"), floor=floor / 1e6)
    line_panels({k: v for k, v in runs.items() if k[0] != "off"}, rhos,
                lambda rs: [y / 1e3 for y in per_wave(rs, "prefill_tokens")], "prefill tokens (thousands)",
                "Tokens prefilled per turn", os.path.join(out, "by_turn"), args.max_turn,
                floor_for=lambda rho: [y / 1e3 for y in new_by_wave])
    line_panels(runs, rhos, lambda rs: [median(x) for x in zip(*[r["summary"]["median_first_token_seconds_by_wave"] for r in rs])],
                "median time to first token (s)", "Median time to first token, per turn",
                os.path.join(out, "ttft"), args.max_turn)
    bar_chart(runs, rhos, lambda rs: median([r["summary"]["output_tokens_per_second"] for r in rs]),
              "output tokens per second", "Output tokens per second", lambda y: f"{y:,.0f}",
              os.path.join(out, "throughput"))

    summary = table(runs, rhos, floor)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"{workload}; new tokens only = {floor:,}\n\n{summary}\n")
    print(f"{workload}; new tokens only = {floor:,}\n\n{summary}\n\nwrote {out}/")


if __name__ == "__main__":
    main()
