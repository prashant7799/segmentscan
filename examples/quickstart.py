"""The README quickstart, as a script. Synthetic data only."""

import numpy as np
import pandas as pd

from segmentscan import scan

# 6,000 synthetic website visits with three category columns.
rng = np.random.default_rng(42)
n = 6000
df = pd.DataFrame(
    {
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "country": rng.choice(["IN", "UK", "NZ", "DE"], n),
        "referrer": rng.choice(["search", "social", "newsletter"], n),
    }
)
# Everyone signs up 30% of the time, except tablet visitors from the newsletter: 50%.
special = (df["device"] == "tablet") & (df["referrer"] == "newsletter")
df["signed_up"] = (rng.random(n) < np.where(special, 0.50, 0.30)).astype(int)

result = scan(
    df,
    dims=["device", "country", "referrer"],
    outcome="signed_up",
    threshold=0.30,  # the baseline rate, chosen before looking
    min_lift=0.05,  # only care about segments at least 5 points better
)
print(result)
