import pandas as pd
import numpy as np

def generate_noisy_dummy_data(
    output_path="data/dummy_mwsel.csv"
):
    np.random.seed(42)

    rows = []
    start = pd.Timestamp.utcnow().floor("min")

    for run in [1, 2]:
        mw = 0
        run_start = start + pd.Timedelta(minutes=(run - 1) * 120)
        time_index = pd.date_range(
            start=run_start,
            periods=120,
            freq="1min",
            tz="UTC"
        )

        for i, t in enumerate(time_index):
            step = i % 30
            
            # ramp up/down
            if step < 15:
                mw += np.random.uniform(0.5, 2.0)
            else:
                mw -= np.random.uniform(1.5, 3.0)
                mw = max(mw, 0)

            # --- random spike for noise ---
            noise = np.random.normal(0, 0.5)  # avg 0, std .5
            mw_noisy = max(mw + noise, 0)

            # --- partial range ramp up/down ---
            if run == 2 and 30 <= i < 60:
                mw_noisy *= np.random.uniform(0.3, 0.8)  # incomplete up/down for testing

            rows.append({
                "Local_time": t,
                "Unit": "U1",
                "RunNumber": run,
                "MWSEL": round(mw_noisy, 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("Noisy dummy data created:", df.shape)
    return df

if __name__=="__main__":
    generate_noisy_dummy_data()
