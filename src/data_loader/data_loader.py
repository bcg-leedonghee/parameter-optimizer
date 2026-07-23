# isort:imports-stdlib

# isort:imports-thirdparty
import pandas as pd

# isort:imports-firstparty
from project_paths import DATA_PATH_BY_VERSION, DATA_PATH_TRACE_BY_VERSION
import polars as pl
# isort:import-localfolder


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, version: str, engine='pandas') -> pd.DataFrame:
        path = DATA_PATH_BY_VERSION[version]
        if ".parquet" in str(path):
            if engine=='pandas':
                data = pd.read_parquet(path)
            elif engine=='polars' :
                data = pl.read_parquet(path)
        elif ".csv" in str(path):
            data = pd.read_csv(path)
        else:
            raise ValueError()
        return data

    def load_trace(self, version: str) -> pd.DataFrame:
        path = DATA_PATH_TRACE_BY_VERSION[version]

        data = pd.read_parquet(path)

        return data
