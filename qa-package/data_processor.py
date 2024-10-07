import os 
import polars as pl 
import time

class DataProcessor:
    # Dictionary to determine which tables are SCDM tables 
    VARIABLE_TO_MODEL_MAPPING = {
        "px_codetype": "PROCEDURE",
        "dtimpute": "DEATH",
        "race": "DEMOGRAPHIC",
        "dx_codetype": "DIAGNOSIS",
        "rx_codetype": "DISPENSING",
        "discharge_disposition": "ENCOUNTER",
        "chart": "ENROLLMENT",
        "facility_location": "FACILITY",
        "fast_ind": "LAB",
        "birth_type": "MIL",
        "specialty": "PROVIDER"
    }

    def __init__(self, gui):
        self.gui = gui

    # Loop through and load each file based on what it is
    def load_csv_files(self, selected_files:list) -> dict:
        datasets = {}
        for file in selected_files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            file_ext = os.path.splitext(os.path.basename(file))[1]
            if file_ext == '.csv':
                df = pl.scan_csv(file)
            # TO-DO: Implement parquet functionality 
            elif file_ext == '.parquet':
                df = pl.scan_parquet(file)
            df_cols = [z.lower() for z in df.collect_schema().names()]            
            self.gui.append_log_message(f"{file_name} loaded successfully")
            dataset_name = 'UNKNOWN_DATASET'
            for variable_name, dataset in self.VARIABLE_TO_MODEL_MAPPING.items():
                if variable_name in df_cols:
                    dataset_name = dataset
                    break
            datasets[dataset_name] = df
            if dataset_name == 'UNKNOWN_DATASET':
                self.gui.append_log_message(f"{file_name} is not a valid SCDM table. This table will not be included")
        return datasets 
    
    # Implement aggregation for encounter table 
    def encounter_module(self, datasets):
        self.gui.append_log_message("\n" + f"Running Encounter module")
        encounter_df=datasets['ENCOUNTER']

        t1=time.time()

        df_1 = (
             encounter_df
             .group_by(["EncType", "Admitting_Source", "Discharge_Disposition","Discharge_Status", pl.concat_str(pl.col("ADate").str.to_date("%m/%d/%Y").dt.year(), pl.col("ADate").str.to_date("%m/%d/%Y").dt.month(), separator='-').alias("YearMonth"), "DDate"])
             .agg(pl.len().alias("count"))
            )

        temp = df_1.group_by("EncType","DDate").agg(pl.sum("count")).collect()

        t2=time.time()

        self.gui.append_log_message("\n" + f"Encounter module completed in {round(t2-t1,2)} seconds")