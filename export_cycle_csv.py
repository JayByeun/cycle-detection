import sqlite3
import pandas as pd

conn = sqlite3.connect("cycle_results.db")

df = pd.read_sql_query("SELECT * FROM cycle_events", conn)

df.to_csv("cycle_results.csv", index=False)

print("cycle_results.csv created")
