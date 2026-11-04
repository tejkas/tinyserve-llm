"""
Run a benchmark grid: one runner process per (pressure, caching, repeat), each writing one JSON file.

    python -m benchmarks.driver --engine ours --model /path/to/checkpoint --trace benchmarks/data/trace_llama.json \
        --conversations 300 --rhos full,1,2 --caching on,off --repeats 3 --out-dir benchmarks/results/llama

Each run is its own process because a runner holds one engine (and its prefix cache) for all of a
replay's waves, and because each engine lives in its own environment (--python picks the
interpreter). Finished runs are skipped, so a grid cut short by a job time limit resumes where it
stopped when run again.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["ours", "nano"], required=True)
    parser.add_argument("--python", default=sys.executable, help="interpreter of the engine's environment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--conversations", type=int, required=True)
    parser.add_argument("--rhos", default="full,1,2", help="comma-separated pressures; 'full' is all free memory")
    parser.add_argument("--caching", default="on", help="on, off or on,off (our engine only; nano always caches)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--tag", default="", help="added to file names, e.g. 'calibration'")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cachings = args.caching.split(",") if args.engine == "ours" else ["on"]
    runs = [(rho, caching, r) for rho in args.rhos.split(",") for caching in cachings for r in range(args.repeats)]
    for i, (rho, caching, repeat) in enumerate(runs, 1):
        name = f"{args.engine}{'_' + args.tag if args.tag else ''}_rho{rho}_cache{caching}_r{repeat}.json"
        out = os.path.join(args.out_dir, name)
        if os.path.exists(out):
            print(f"[{i}/{len(runs)}] {name}: done already", flush=True)
            continue
        command = [args.python, "-m", f"benchmarks.run_{args.engine}", "--model", args.model, "--trace", args.trace,
                   "--conversations", str(args.conversations), "--rho", rho, "--seed", str(repeat), "--out", out + ".tmp"]
        if args.engine == "ours":
            command += ["--caching", caching]
        if args.max_waves is not None:
            command += ["--max-waves", str(args.max_waves)]
        print(f"[{i}/{len(runs)}] {name}", flush=True)
        start = time.time()
        subprocess.run(command, check=True)
        os.replace(out + ".tmp", out)  # only a finished run counts as done
        print(f"[{i}/{len(runs)}] {name}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
