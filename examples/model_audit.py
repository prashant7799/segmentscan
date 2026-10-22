"""Where is a classifier less accurate than 90%? Synthetic predictions.

Each row is one prediction. ``correct`` is 1 when the model was right.
The model is right about 93% of the time overall, but only about 80% of
the time for customers under 25 who signed up through the app.

Two things to notice:

* ``alternative="less"`` looks for cells BELOW the bar.
* ``age`` and ``tenure_months`` are numbers. ``bins=`` cuts them into
  quantile bins, with edges learned on the explore split only.

Run:
    python examples/model_audit.py
"""

import numpy as np
import pandas as pd

from segmentscan import scan


def make_predictions(seed: int = 11, n: int = 30_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 75, n),
            "tenure_months": rng.integers(0, 60, n),
            "channel": rng.choice(["web", "app", "store"], n),
            "segment": rng.choice(["consumer", "smb", "enterprise"], n),
        }
    )
    accuracy = np.full(n, 0.93)
    weak = (df["age"] < 25) & (df["channel"] == "app")
    accuracy[weak.to_numpy()] = 0.80
    df["correct"] = (rng.random(n) < accuracy).astype(int)
    return df


def main() -> None:
    df = make_predictions()
    print(f"{len(df)} predictions, overall accuracy {df['correct'].mean():.3f}\n")

    result = scan(
        df,
        dims=["age", "tenure_months", "channel", "segment"],
        outcome="correct",
        threshold=0.90,  # the accuracy we promised
        alternative="less",
        bins={"age": 6, "tenure_months": 3},
        q=0.05,
    )
    print(result)
    print("\nAge bin edges (fitted on explore rows):", result.bin_edges["age"][1:-1])


if __name__ == "__main__":
    main()
