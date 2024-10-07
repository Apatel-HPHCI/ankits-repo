import tkinter as tk 
from gui import GUI 

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sentinel QA Package")
    app = GUI(root)
    root.mainloop()

import time
t1=time.time()
import polars as pl 
df = pl.scan_csv("U:\\mscdm\\v8.2.0\\Truven\\etl-24\\sample_1pct\\encounter.csv", try_parse_dates=True)
df_1 = (
             df
             .group_by(["EncType", "Admitting_Source", "Discharge_Disposition","Discharge_Status", pl.concat_str(pl.col("ADate").str.to_date("%m/%d/%Y").dt.year(), pl.col("ADate").str.to_date("%m/%d/%Y").dt.month(), separator='-').alias("YearMonth"), "DDate"])
             .agg(pl.len().alias("count"))
            )

temp = df_1.group_by("EncType","DDate").agg(pl.sum("count")).collect()
t2= time.time()
print(temp)
print(round(t2-t1,2), 'seconds')