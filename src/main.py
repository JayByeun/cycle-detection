import pandas as pd
from cycle_detector import detect_cycles

df = pd.read_csv("data/dummy_timeseries.csv", parse_dates=["Local_Time"])

result = detect_cycles(df)

summary = result.groupby(["Unit", "RunNumber"]).agg(
    QtyFullCycles=("PartialCycleType", lambda x: (x == 1.0).sum()),
    QtyPartial_75=("PartialCycleType", lambda x: (x == 0.75).sum()),
    QtyPartial_50=("PartialCycleType", lambda x: (x == 0.5).sum()),
    QtyPartial_10=("PartialCycleType", lambda x: (x == 0.1).sum()),
)

print(summary)