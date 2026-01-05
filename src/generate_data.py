import pandas as pd
import numpy as np


def generate_dummy_data(
    output_path="data/dummy_mwsel.scv"
):
    np.random.seed(42)

    rows = []
    start = pd.Timestamp.utcnow().floor("min")

    for run in [1, 2]:
        mw = 0
        run_start = start + pd.Timedelta(minutes=(run - 1) * 120) # create 120 data with interval 1 min
        time_index = pd.date_range(
            start=run_start,
            periods=120,
            freq="1min",
            tz="UTC"
        )
        
        for t in time_index:
            step = len(rows) % 30
            if step < 15:
                mw += np.random.uniform(0.5, 2.0) # ramp up
            else:
                mw -= np.random.uniform(0.5, 2.0) # drop

            mw = max(mw, 0)

            rows.append({
                "Local_time": t,
                "Unit": "U1",
                "RunNumber": run,
                "MWSEL": round(mw, 2)
            }) # output

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("Dummy data created:", df.shape)
    return df

if __name__=="__main__":
    generate_dummy_data()
