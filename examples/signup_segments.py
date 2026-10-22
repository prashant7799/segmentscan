"""Which customer segments convert above baseline? Synthetic visits, clustered by user.

One user can visit many times, and a keen user signs up more often on
every visit. So visits from the same user are not independent.

The script runs the same scan twice:

1. treating every visit as independent (wrong for this data), and
2. counting users, with cluster="user_id".

Only one real effect is planted: desktop visits from the newsletter sign up
12 points more often. Country and plan page have NO real effect.

Run:
    python examples/signup_segments.py
    python examples/signup_segments.py --save-csv data/signups.csv   # for the CLI demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from segmentscan import scan


def make_signups(seed: int = 7, n_users: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Fixed per user.
    keenness = rng.beta(0.9, 2.1, n_users)  # each user's own sign-up rate, mean 0.30
    country = rng.choice(["IN", "UK", "NZ", "DE", "US"], n_users)
    plan_page = rng.choice(["basic", "team", "enterprise"], n_users)
    visits_per_user = rng.integers(1, 9, n_users)  # 1 to 8 visits

    # Changes from visit to visit.
    user = np.repeat(np.arange(n_users), visits_per_user)
    device = rng.choice(["mobile", "desktop", "tablet"], len(user))
    referrer = rng.choice(["search", "social", "newsletter"], len(user))

    rate = keenness[user] + 0.12 * ((device == "desktop") & (referrer == "newsletter"))
    signed_up = rng.random(len(user)) < np.clip(rate, 0, 1)

    return pd.DataFrame(
        {
            "user_id": user,
            "country": country[user],
            "plan_page": plan_page[user],
            "device": device,
            "referrer": referrer,
            "signed_up": signed_up.astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-csv", help="also write the synthetic table to this CSV path")
    args = parser.parse_args()

    df = make_signups()
    print(f"{len(df)} visits from {df['user_id'].nunique()} users\n")

    dims = ["country", "plan_page", "device", "referrer"]
    settings = dict(threshold=0.30, min_lift=0.03, q=0.05)

    print("1) Every visit counted as independent (wrong here):")
    by_visit = scan(df, dims, "signed_up", **settings)
    print(by_visit)
    print()

    print('2) Each user counted once per cell (cluster="user_id"):')
    by_user = scan(df, dims, "signed_up", cluster="user_id", **settings)
    print(by_user)
    print(f"\nDesign effect on the explore split: {by_user.design_effect:.2f}")

    if args.save_csv:
        path = Path(args.save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
