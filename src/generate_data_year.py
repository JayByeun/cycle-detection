import pandas as pd
import numpy as np

def generate_noisy_yearly_dummy_data(
    output_path="data/dummy_mwsel_yearly.csv"
):
    np.random.seed(42)
    rows = []

    start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")  # start a yaer
    num_runs = 12  # run per month

    for run in range(1, num_runs+1):
        mw = 0
        run_start = start + pd.DateOffset(months=run-1)
        time_index = pd.date_range(
            start=run_start,
            periods=240,  # 240min
            freq="1min",
            tz="UTC"
        )

        for i, t in enumerate(time_index):
            step = i % 30
            cycle_step = (i // 30) % 3

            # ramp up/down
            if step < 15:
                mw += np.random.uniform(0.5, 2.0)
            else:
                mw -= np.random.uniform(1.5, 3.0)
                mw = max(mw, 0)

            if cycle_step == 1:
                mw *= np.random.uniform(1.0, 1.2)
            elif cycle_step == 2:
                mw *= np.random.uniform(0.8, 1.0)

            # --- random noise ---
            noise = np.random.normal(0, 0.5)
            mw_noisy = max(mw + noise, 0)

            # --- partial range ramp up/down ---
            if run == 2 and 30 <= i < 60:
                mw_noisy *= np.random.uniform(0.3, 0.8)

            rows.append({
                "Local_time": t,
                "Unit": "U1",
                "RunNumber": run,
                "StartTime": run_start,
                "MWSEL": round(mw_noisy, 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("Noisy yearly dummy data created:", df.shape)
    return df

if __name__=="__main__":
    generate_noisy_yearly_dummy_data()
