# isort:imports-stdlib
import json
import re
from typing import Dict, List

# isort:imports-thirdparty
import pandas as pd

# isort:imports-firstparty
from project_paths import COLUMN_MAPPING_PATH_BY_TYPE_PROCESS

# isort:import-localfolder

SUPPORTED_COLUMN_LIST = [
    "Mixing",
    "Coating",
    "Slitting",
    "Roll_Pressing",
    "Temp",
    "Humidity",
    "IQC",
    "Winding",
    "Assembly",
]


class ProcessMapper:
    def __init__(self):
        pass

    def map(self, data: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
        data = (
            data.assign(
                Mixing=lambda x: x[target_column_name].str.contains("Mixing"))
            .assign(Coating=lambda x: x[target_column_name].str.contains("Coating"))
            .assign(Slitting=lambda x: x[target_column_name].str.contains("Slitting"))
            .assign(
                Roll_Pressing=lambda x: x[target_column_name].str.contains(
                    "Roll Pressing"
                )
            )
            .assign(Winding=lambda x: x[target_column_name].str.contains("Winding"))
            .assign(Assembly=lambda x: x[target_column_name].str.contains("Assembly"))
            .assign(Temp=lambda x: x[target_column_name].str.contains("Temp"))
            .assign(Humidity=lambda x: x[target_column_name].str.contains("Humidity"))
            .assign(IQC=lambda x: x[target_column_name].str.contains("IQC"))
        )

        data["process"] = data[SUPPORTED_COLUMN_LIST].apply(
            lambda row: ", ".join(row.index[row]), axis=1
        )

        data = data.drop(SUPPORTED_COLUMN_LIST, axis=1)
        return data
