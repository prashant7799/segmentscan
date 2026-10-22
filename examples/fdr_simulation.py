"""Check the false-discovery promise by simulation. Synthetic data only.

Part 1 - independent rows. Three overlapping dims (3 x 3 x 2 levels, 47
cells), 2,000 rows per run. Either nothing is special (global null), or the
cell a=a0 & b=b0 has rate 0.40 instead of 0.30.

Part 2 - clustered rows with no real effect: 400 users x 10 rows. Each user
has their own rate (intraclass correlation 1/3). Counting rows as if they
were independent is compared with counting users (cluster="user_id").

Part 3 - harder settings with no real effect, BH only:
  * every rate sits exactly on the bar (threshold 0.25 + min_lift 0.05);
  * alternative="less";
  * five yes/no dims, 130 heavily overlapping cells (half the runs);
  * users with 1 to 15 rows each, counted with cluster="user_id".

A "false discovery" is a discovered cell whose true rate is at or below the
bar. FDR is the average share of false discoveries among discoveries
(0 when there are none).

Run (about a minute):
    python examples/fdr_simulation.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from segmentscan import scan

LEVELS = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"], "c": ["c0", "c1"]}
BASE = 0.30


def make_table(n: int, seed: int, planted: float | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({dim: rng.choice(levels, n) for dim, levels in LEVELS.items()})
    rate = np.full(n, BASE)
    if planted is not None:
        rate[((df["a"] == "a0") & (df["b"] == "b0")).to_numpy()] = planted
    df["y"] = (rng.random(n) < rate).astype(int)
    return df


def true_rate(dims, levels, planted: float | None) -> float:
    """Average rate over the grid points inside the cell (dims are uniform)."""
    rates = []
    for combo in itertools.product(*LEVELS.values()):
        point = dict(zip(LEVELS, combo, strict=True))
        if all(point[d] == v for d, v in zip(dims, levels, strict=True)):
            is_planted = planted is not None and point["a"] == "a0" and point["b"] == "b0"
            rates.append(planted if is_planted else BASE)
    return float(np.mean(rates))


def part_one(runs: int, q: float) -> None:
    print(f"Part 1: independent rows, {runs} runs per line, q = {q}")
    print(f"{'scenario':<14}{'method':<8}{'FDR':>7}{'any false':>11}{'planted found':>15}")
    for scenario, planted in (("global null", None), ("one planted", 0.40)):
        for method in ("bh", "by", "holm"):
            shares, any_false, found_planted = [], 0, 0
            for run in range(runs):
                df = make_table(2000, seed=10_000 + run, planted=planted)
                result = scan(df, ["a", "b", "c"], "y", BASE, q=q, method=method, seed=run)
                found = result.discoveries
                n_false = sum(
                    true_rate(d, v, planted) <= result.bar
                    for d, v in zip(found["dims"], found["levels"], strict=True)
                )
                shares.append(n_false / max(len(found), 1))
                any_false += n_false > 0
                found_planted += "a=a0 & b=b0" in set(found["cell"])
            planted_text = "-" if planted is None else f"{found_planted / runs:.2f}"
            print(
                f"{scenario:<14}{method:<8}{np.mean(shares):>7.3f}"
                f"{any_false / runs:>11.3f}{planted_text:>15}"
            )


def make_clustered(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_users, rows_each = 400, 10
    user_rate = rng.beta(0.6, 1.4, n_users)  # mean 0.30, intraclass correlation 1/3
    region = rng.choice(["north", "south", "east", "west"], n_users)
    plan = rng.choice(["free", "basic", "pro"], n_users)
    user = np.repeat(np.arange(n_users), rows_each)
    df = pd.DataFrame(
        {
            "user_id": user,
            "region": region[user],
            "plan": plan[user],
            "device": rng.choice(["ios", "android", "web"], len(user)),
        }
    )
    df["y"] = (rng.random(len(user)) < user_rate[user]).astype(int)
    return df


def part_two(runs: int, q: float) -> None:
    print(f"\nPart 2: clustered rows, no real effect, {runs} runs, q = {q}")
    dims = ["region", "plan", "device"]
    by_rows, by_users = 0, 0
    for run in range(runs):
        df = make_clustered(seed=20_000 + run)
        rows = scan(df, dims, "y", BASE, q=q, min_n=10, seed=run)
        users = scan(df, dims, "y", BASE, q=q, min_n=10, seed=run, cluster="user_id")
        by_rows += len(rows.discoveries) > 0
        by_users += len(users.discoveries) > 0
    print(f"rows treated as independent : FDR {by_rows / runs:.3f}")
    print(f'cluster="user_id"           : FDR {by_users / runs:.3f}')


def make_binary(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"d{i}": rng.choice(["x", "z"], 2000) for i in range(5)})
    df["y"] = (rng.random(2000) < BASE).astype(int)
    return df


def make_unequal_clusters(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_users = 500
    user_rate = rng.beta(0.6, 1.4, n_users)
    sizes = rng.integers(1, 16, n_users)  # 1 to 15 rows per user
    region = rng.choice(["north", "south", "east", "west"], n_users)
    user = np.repeat(np.arange(n_users), sizes)
    df = pd.DataFrame(
        {
            "user_id": user,
            "region": region[user],
            "device": rng.choice(["ios", "android", "web"], len(user)),
        }
    )
    df["y"] = (rng.random(len(user)) < user_rate[user]).astype(int)
    return df


def share_with_discoveries(make, runs: int, dims: list[str], **options) -> float:
    """With no real effect every discovery is false, so FDR = P(any discovery)."""
    hits = 0
    for run in range(runs):
        result = scan(make(run), dims, "y", seed=run, **options)
        hits += len(result.discoveries) > 0
    return hits / runs


def part_three(runs: int, q: float) -> None:
    print(f"\nPart 3: harder settings, no real effect, BH, q = {q}")
    abc = ["a", "b", "c"]
    settings = [
        (
            "rate exactly on the bar (min_lift)",
            lambda run: make_table(2000, seed=30_000 + run, planted=None),
            runs,
            abc,
            dict(threshold=0.25, min_lift=0.05),
        ),
        (
            'alternative="less"',
            lambda run: make_table(2000, seed=40_000 + run, planted=None),
            runs,
            abc,
            dict(threshold=BASE, alternative="less"),
        ),
        (
            "130 overlapping cells",
            lambda run: make_binary(seed=50_000 + run),
            runs // 2,
            [f"d{i}" for i in range(5)],
            dict(threshold=BASE),
        ),
        (
            "unequal clusters, by user",
            lambda run: make_unequal_clusters(seed=60_000 + run),
            runs,
            ["region", "device"],
            dict(threshold=BASE, cluster="user_id", min_n=10),
        ),
    ]
    for name, make, n_runs, dims, options in settings:
        fdr = share_with_discoveries(make, n_runs, dims, q=q, **options)
        print(f"{name:<36}: FDR {fdr:.3f} ({n_runs} runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--q", type=float, default=0.10)
    args = parser.parse_args()
    part_one(args.runs, args.q)
    part_two(args.runs // 2, args.q)
    part_three(args.runs, args.q)


if __name__ == "__main__":
    main()
