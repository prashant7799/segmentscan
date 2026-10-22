"""Which groups in the UCI Adult census data earn more than $50K? Bring your own copy.

This script does not download anything. Get the data yourself from the UCI
Machine Learning Repository ("Adult", donated 1996, licence CC BY 4.0):

    https://archive.ics.uci.edu/dataset/2/adult

Unzip it and pass the path of ``adult.data`` (no header row, 15 columns):

    python examples/uci_adult.py path/to/adult.data

About 24% of rows in adult.data have income >50K, so the threshold is 0.24.
Strictly, that number comes from the same rows, so it is not fixed in
advance. With 32,561 rows the overall rate is very precise, and min_lift=0.10
is far larger than its wobble. With your own data, take the threshold from
earlier data or from a target instead.

This is a description of who earns more in a 1994 census extract. It is
not a causal claim about education, work or marriage.
"""

from __future__ import annotations

import argparse

import pandas as pd

from segmentscan import scan

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]  # fmt: skip


def load_adult(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True, na_values="?")
    df = df.dropna(subset=["income"])
    # adult.test writes ">50K." with a trailing dot; adult.data writes ">50K".
    df["high_income"] = df["income"].str.rstrip(".").eq(">50K").astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="path to adult.data (or adult.test)")
    args = parser.parse_args()

    df = load_adult(args.path)
    print(f"{len(df)} rows, share with income >50K: {df['high_income'].mean():.3f}\n")

    result = scan(
        df,
        dims=["education", "marital_status", "occupation", "hours_per_week"],
        outcome="high_income",
        threshold=0.24,
        min_lift=0.10,  # only groups at least 10 points above the base rate
        bins={"hours_per_week": [35, 45]},  # part time, around 40 hours, long weeks
        method="by",
        min_n=100,
    )
    print(result.summary(max_rows=15))


if __name__ == "__main__":
    main()
