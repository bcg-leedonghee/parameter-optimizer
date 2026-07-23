# isort:imports-stdlib
import json
import os
import re
from typing import Dict, List

# isort:imports-thirdparty
import pandas as pd

# isort:imports-firstparty
from project_paths import COLUMN_MAPPING_PATH_BY_TYPE_PROCESS

# isort:import-localfolder


class FeatureMapper:
    def __init__(self):
        pass

    # def _normalize_code(self, x):
    #     x = str(x).strip().upper()

    #     m = re.match(r"^([A-Z]+)0*(\d+)$", x)
    #     if m :
    #         return f"{m.group(1)}{int(m.group(2))}"
    #     return x

    def _normalize_code(self, x):
        x = str(x).strip().upper()

        return re.sub(r"^([A-Z]+)0+", r"\1", x)

    def _get_code(self, cols_list: List, type_process: str) -> List:
        if "Electrode" in type_process:
            cols_list_code = [
                x.split("_")[4].upper() if len(x.split("_")) > 4 else ""
                for x in cols_list
            ]
            # cols_list_code = [self._pad_code(x) for x in cols_list_code]
            # cols_list_mapped = [map_dict.get(x) for x in cols_list_code if map_dict.get(x) is not None]
            print("cols_list_Code")
            print(cols_list_code)
        elif "PV_" in type_process:
            cols_list_code = [x.split("_")[3].upper() for x in cols_list]
            # cols_list_mapped = [map_dict.get(x) for x in cols_list_code if map_dict.get(x)]
        return cols_list_code

    def _get_mapped_name(self, feature: str, map_dict: Dict):
        mapped_value = map_dict.get(feature)
        if mapped_value is None:
            mapped_value = ""
        return mapped_value

    def map(self, cols_list: List, type_process: str) -> List:
        """
        COLUMN_MAPPING_PATH_BY_TYPE_PROCESS = {
            'PV_Electrode' : "/data01/Documents/0_hw/data/mapping/PV_Electrode.json",
            'PV_Assembly': "/data01/Documents/0_hw/data/mapping/PV_Assembly.json",
            'PV_Winding' : '/data01/Documents/0_hw/data/mapping/PV_Winding.json',
            'SV_Mixing' : '/data01/Documents/0_hw/data/mapping/SV_Mixing.json',
            'SV_Coating' : '/data01/Documents/0_hw/data/mapping/SV_Coating.json',
            'SV_Roll Pressing' : '/data01/Documents/0_hw/data/mapping/SV_Roll Pressing.json',
            'SV_Slitting' : '/data01/Documents/0_hw/data/mapping/SV_Slitting.json',
            'SV_Winding' : '/data01/Documents/0_hw/data/mapping/SV_Winding.json',
            'SV_Assembly' : '/data01/Documents/0_hw/data/mapping/SV_Assembly.json',
            'SV_Washing' : '/data01/Documents/0_hw/data/mapping/SV_Washing.json',
        }

        """
        map_path = COLUMN_MAPPING_PATH_BY_TYPE_PROCESS[type_process]

        # if type_process == 'PV_Electrode' :
        #     folder_path = '/data01/Documents/0_hw/data/mapping/keys'
        #     file_list = os.listdir(folder_path)
        #     map_dict_original = []
        #     for file_name in file_list :
        #         tmp = pd.read_csv(f"{folder_path}/{file_name}")
        #         map_dict_original.append(tmp)
        #     map_dict_original = pd.concat(map_dict_original, axis=0)
        #     map_dict = dict(zip(map_dict_original['name'], map_dict_original['displayName']))
        #     print(map_dict)
        #     with open('/data01/Documents/0_hw/data/mapping/PV_Electrode_v2.json', 'w', encoding='utf-8') as f :
        #         json.dump(map_dict, f, ensure_ascii=False, indent=4)

        # else :

        with open(map_path, "r", encoding="utf-8") as f:
            map_dict_original = json.load(f)
            if "Electrode" in type_process:
                # map_dict_7 = {self._pad_code_7(k):v for k, v in map_dict_original.items()}
                # map_dict_6 = {self._pad_code_6(k):v for k, v in map_dict_original.items()}

                # map_dict_7 = {k.upper(): v for k, v in map_dict_7.items()}
                # map_dict_6 = {k.upper(): v for k, v in map_dict_6.items()}
                # print('map_dict_7')
                # print(map_dict_7)
                # print('map_dict_6')
                # print(map_dict_6)
                map_dict = map_dict_original
                map_dict = {k.upper(): v for k, v in map_dict.items()}
            else:
                map_dict = map_dict_original
                map_dict = {k.upper(): v for k, v in map_dict.items()}

        # print(map_dict)

        if "Electrode" in type_process:
            cols_list_code = [
                x.split("_")[4].upper() if len(x.split("_")) > 4 else ""
                for x in cols_list
            ]
            # cols_list_mapped = [map_dict.get(x) for x in cols_list_code if map_dict.get(x) is not None]
            cols_list_mapped = [
                self._get_mapped_name(self._normalize_code(x), map_dict)
                for x in cols_list_code
            ]
            # cols_list_mapped_6 = [self._get_mapped_name(x, map_dict_6) for x in cols_list_code]

            # cols_list_mapped = [x if x!= "" else y for x, y in zip(cols_list_mapped_7, cols_list_mapped_6)]
        else:
            # print(cols_list_code)
            cols_list_code = [
                x.split("_")[3].upper() if len(x.split("_")) > 3 else ""
                for x in cols_list
            ]
            # cols_list_mapped = [map_dict.get(x) for x in cols_list_code if map_dict.get(x) is not None]
            cols_list_mapped = [
                self._get_mapped_name(x, map_dict) for x in cols_list_code
            ]

        # elif 'SV_' in type_process :
        #     cols_list_code = [x.split('_')[1].upper() for x in cols_list]
        #     cols_list_mapped = [map_dict.get(x) for x in cols_list_code if map_dict.get(x) is not None]
        # else :
        #     raise ValueError()

        return cols_list_mapped

    def remap_column_name(self, data: pd.DataFrame) -> pd.DataFrame:

        data = self._remap_column_name(
            data=data, key1="X_Cathode", key2="PV_Electrode")
        data = self._remap_column_name(
            data=data, key1="X_Anode", key2="PV_Electrode")
        data = self._remap_column_name(
            data=data, key1="X_PV", key2="PV_Assembly")
        data = self._remap_column_name(
            data=data, key1="X_PV", key2="PV_Winding")

        return data

    def _remap_column_name(
        self, data: pd.DataFrame, key1: str, key2: str
    ) -> pd.DataFrame:

        list_cols_pv = [x for x in data.columns if key1 in x]

        code_original = self._get_code(
            cols_list=list_cols_pv, type_process=key2)
        code_mapped = self.map(cols_list=list_cols_pv, type_process=key2)

        for code, name in zip(code_original, code_mapped):
            data.columns = [x.replace(code, name) for x in data.columns]

        return data
