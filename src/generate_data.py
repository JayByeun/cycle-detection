import pandas as pd
import numpy as np

np.random.seed(42)

rows = []
start_time = pd.Timestamp("2024-01-01 00:00:00")

for run in [1, 2]:
    mw = 0
    for i in range(120):
        if i % 30 < 15:
            mw += np.random.uniform(0.5, 2.0) # ramp up
        else:
            mw -= np.random.uniform(0.5, 2.0) # drop
        
        mw = max(mw, 0)

        rows.append({
            "Local_time": start_time + pd.Timedelta(minutes=i),
            "Unit": "U1",
            "RunNumber": run,
            "MWSEL": round(mw, 2)
        })

df = pd.DataFrame(rows)
df.to_csv("data/dummy_mwsel.csv", index=False)

print("Dummy data created:", df.shape)
