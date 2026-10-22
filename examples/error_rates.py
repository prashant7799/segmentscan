"""Which device / browser / app-version combos have a higher error rate? Synthetic sessions.

A QA team has an error budget of 2% of sessions. Two problems are planted:

* Android on app version 5.3 fails 4.0% of the time.
* Safari on the web client fails 3.5% of the time.

Everything else fails 2.0% of the time.

The scan uses Benjamini-Yekutieli (valid under any dependence), a minimum
lift of half a point, and a third "confirm" split to re-test discoveries.

Run:
    python examples/error_rates.py
"""

import numpy as np
import pandas as pd

from segmentscan import scan


def make_sessions(seed: int = 3, n: int = 240_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "client": rng.choice(["ios", "android", "web"], n),
            "browser": rng.choice(["chrome", "safari", "firefox", "edge"], n),
            "app_version": rng.choice(["5.1", "5.2", "5.3"], n),
            "region": rng.choice(["eu", "us", "apac"], n),
        }
    )
    rate = np.full(n, 0.020)
    rate[(df["client"] == "android") & (df["app_version"] == "5.3")] = 0.040
    rate[(df["client"] == "web") & (df["browser"] == "safari")] = 0.035
    df["error"] = (rng.random(n) < rate).astype(int)
    return df


def main() -> None:
    df = make_sessions()
    print(f"{len(df)} sessions, overall error rate {df['error'].mean():.4f}\n")

    result = scan(
        df,
        dims=["client", "browser", "app_version", "region"],
        outcome="error",
        threshold=0.02,  # the error budget
        min_lift=0.005,  # ignore cells less than half a point over budget
        method="by",
        holdout=0.4,
        confirm=0.2,
        min_n=200,
    )
    print(result)


if __name__ == "__main__":
    main()
