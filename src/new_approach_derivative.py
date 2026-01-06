import pandas as pd
import numpy as np
import sqlite3

def detect_cycles_diff(
    csv_path="data/dummy_mwsel_noise.csv",
    db_path="cycle_results_new.db",
    smoothing_window=5,   # rolling mean window to remove noise (using pre/next 5 mean)
    min_cycle_len=3       # min peak/valley sample length (ignore a cycle if it has under 3 peak/valley)
):
    df = pd.read_csv(csv_path, parse_dates=["Local_time"])
    df = df.sort_values(["Unit", "RunNumber", "Local_time"])
    
    g = df.groupby(["Unit", "RunNumber"])

    # --- 1. remove noise: rolling mean ---
    df['mw_rolling'] = g['MWSEL'].transform(lambda x: x.rolling(smoothing_window, min_periods=1).mean())
    
    # --- 2. use derivative to calculate direction ---
    df['diff'] = df.groupby(['Unit','RunNumber'])['mw_rolling'].diff()
    df['direction'] = np.sign(df['diff']).fillna(0)
    
    # --- 3. calculate peak/valley by comparing previous direction ---
    df['prev_direction'] = df.groupby(['Unit','RunNumber'])['direction'].shift(1).fillna(0)
    
    # peak: transition up to down
    df['peak'] = np.where((df['prev_direction'] > 0) & (df['direction'] <= 0) & (df['mw_rolling'] > 0), df['mw_rolling'], np.nan)
    # valley: trainsition down to up
    df['valley'] = np.where((df['prev_direction'] < 0) & (df['direction'] >= 0), df['mw_rolling'], np.nan)
    
    # --- 4. cycle grouping ---
    df['cycle_group'] = df['peak'].notna().cumsum()
    
    # calculate forward fill peak for partial drop
    df['max_group'] = df.groupby(['Unit','RunNumber'])['peak'].ffill()
    
    # --- 5. calculate partial / full cycle ---
    df['partial_drop'] = np.where(
        (df['cycle_group'].shift(-1) > df['cycle_group']) & (df['max_group'] > 0) & (~df['valley'].isna()),
        (df['max_group'] - df['valley']) / df['max_group'],
        np.nan
    )
    
    # cycle type: 1=full, <1=partial (representative 0.75,0.5,0.1)
    def classify_cycle(row):
        # partial_drop insignificant
        if pd.isna(row['partial_drop']):
            return 0
        
        # full cycle
        elif row['valley'] <= row['max_group'] * 0.05:  
            return 1.0
        
        # partial cycle
        elif row['partial_drop'] > 0.75:
            return 0.75
        elif row['partial_drop'] > 0.5:
            return 0.5
        elif row['partial_drop'] > 0.1:
            return 0.1
        
        # insignificant partial
        else:
            return 0

    df['cycle_type'] = df.apply(classify_cycle, axis=1)


    max_partial = 10

    summary_list = []

    for (unit, run), group in df.groupby(['Unit','RunNumber']):
        # qty full cycle
        full_count = (group['cycle_type'] == 1).sum()
        
        # extract partial cycle
        partials = group[(group['cycle_type'] < 1) & (group['cycle_type'] > 0)]
        partials = partials.sort_values('partial_drop', ascending=False).head(max_partial)
        
        # representitive partial qty
        count_75 = (partials['cycle_type'] == 0.75).sum()
        count_50 = (partials['cycle_type'] == 0.5).sum()
        count_10 = (partials['cycle_type'] == 0.1).sum()
        
        # insignificant partial
        count_0 = (partials['cycle_type'] == 0).sum()
        
        summary_list.append({
            'Unit': unit,
            'RunNumber': run,
            'StartTime': group['StartTime'].iloc[0],
            'QtyFullCycles': full_count,
            'QtyPartialCycles_75': count_75,
            'QtyPartialCycles_50': count_50,
            'QtyPartialCycles_10': count_10,
            'QtyPartialCycles_0': count_0
        })

    summary = pd.DataFrame(summary_list)

    
    # --- 6. summary table ---
    # summary = df.groupby(['Unit','RunNumber']).agg(
    #     QtyFullCycles=('cycle_type', lambda x: (x==1).sum()),
    #     QtyPartialCycles_75=('cycle_type', lambda x: (x==0.75).sum()),
    #     QtyPartialCycles_50=('cycle_type', lambda x: (x==0.5).sum()),
    #     QtyPartialCycles_10=('cycle_type', lambda x: (x==0.1).sum()),
    #     QtyPartialCycles_0=('cycle_type', lambda x: (x==0).sum())
    # ).reset_index()
    
    print(summary)
    
    # --- 7. store result in DB ---
    conn = sqlite3.connect(db_path)
    df.to_sql("cycle_events", conn, if_exists="replace", index=False)
    summary.to_sql("cycle_summary", conn, if_exists="replace", index=False)
    conn.close()
    
    return df, summary

if __name__ == "__main__":
    df, summary = detect_cycles_diff()
    print("Cycle detection completed")
