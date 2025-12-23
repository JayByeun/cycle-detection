import pandas as pd
import numpy as np
import sqlite3

df = pd.read_csv("data/dummy_mwsel.csv", parse_dates=["Local_time"])
df = df.sort_values(["Unit", "RunNumber", "Local_time"])

g = df.groupby(["Unit", "RunNumber"])

# Avg() over, stdev() over, rows between 5 pred
df["running_avg"] = g["MWSEL"].expanding().mean().reset_index(level=[0, 1], drop=True)
df["running_std"] = g["MWSEL"].expanding().std().reset_index(level=[0, 1], drop=True)

df["rolling5_avg"] = g["MWSEL"].rolling(6, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
df["rolling5_std"] = g["MWSEL"].rolling(6, min_periods=1).std().reset_index(level=[0, 1], drop=True)

df["rolling5_min"] = g["MWSEL"].rolling(6, min_periods=1).min().reset_index(level=[0, 1], drop=True)
df["rolling5_max"] = g["MWSEL"].rolling(6, min_periods=1).max().reset_index(level=[0, 1], drop=True)

# z-score
df["zscore"] = np.where(
    df["running_std"] == 0,
    0,
    (df["MWSEL"] - df["rolling5_avg"]) / df["running_std"]
)

df["zscore5"] = np.where(
    df["rolling5_std"] == 0,
    0,
    (df["MWSEL"] - df["rolling5_avg"]) / df["rolling5_std"]
)

# z-score normalization (up, down, steady)
df["zscore_norm"] = np.select(
    [
        df["zscore5"] > 0.1,
        df["zscore5"] < -0.05
    ],
    [1, -1],
    default=0
)

# local max/ min (cycle point)
df["next_state"] = g["zscore_norm"].shift(-1)

df["max_mwsel"] = np.where(
    (df["zscore_norm"] == 1) & (df["next_state"] != 1),
    df["rolling5_max"],
    np.nan
)

df["min_mwsel"] = np.where(
    (df["zscore_norm"] == -1) & (df["next_state"] != -1),
    df["rolling5_min"],
    np.nan
)

# cycle grouping
df["cycle_group"] = df["max_mwsel"].notna().groupby([df["Unit"], df["RunNumber"]]).cumsum()

# partial cycle drop
df["max_group"] = g["max_mwsel"].ffill()

df["partial_drop"] = np.where(
    (df["cycle_group"].shift(-1) > df["cycle_group"]) & (df["max_group"] > 20),
    (df["max_group"] - df["min_mwsel"]) / df["max_group"],
    np.nan
)

# partial / full cycle
df["cycle_type"] = np.select(
    [
        (df["partial_drop"] == 1) & (df["MWSEL"] == 0),
        df["partial_drop"] > 0.75,
        df["partial_drop"] > 0.5,
        df["partial_drop"] > 0.1,
        df["partial_drop"] <= 0.1
    ],
    [1.0, 0.75, 0.5, 0.1, 0],
    default=np.nan
)

# summary per run
summary = df.groupby(["Unit", "RunNumber"]).agg(
    QtyFullCycles = ("cycle_type", lambda x: 1+ (x == 1).sum()),
    QtyPartialCycles_75 = ("cycle_type", lambda x: (x == 0.75).sum()),
    QtyPartialCycles_50 = ("cycle_type", lambda x: (x == 0.5).sum()),
    QtyPartialCycles_10 = ("cycle_type", lambda x: (x == 0.1).sum()),
    QtyPartialCycles_0 = ("cycle_type", lambda x: (x == 0).sum()),
).reset_index()

print(summary)


conn = sqlite3.connect("cycle_results.db")

df.to_sql("cycle_events", conn, if_exists="replace", index=False)
summary.to_sql("cycle_summary", conn, if_exists="replace", index=False)

conn.close()

print(df.columns)
print("Saved to cycle_results.db")