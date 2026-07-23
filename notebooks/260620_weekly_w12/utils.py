

import warnings
import sys
from pathlib import Path


def _ensure_project_src_path() -> None:
    current_file_path = Path(__file__).resolve()
    for candidate_repo_root in current_file_path.parents:
        src_path = candidate_repo_root / "src"
        if src_path.exists() and (src_path / "target_columns").exists():
            src_path_string = str(src_path)
            if src_path_string not in sys.path:
                sys.path.insert(0, src_path_string)
            return

    raise ModuleNotFoundError("src/target_columns 경로를 찾지 못했습니다.")


_ensure_project_src_path()

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

import re
import seaborn as sns
import xgboost as xgb
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

from target_columns.cols_small_y import (
    SMALL_Y_DEFECT_ELECTRODE,
    SMALL_Y_CTQ_ELECTRODE,
    SMALL_Y_DEFECT_WINDING,
    SMALL_Y_CTQ_WINDING,
    SMALL_Y_DEFECT_ASSEMBLY,
    SMALL_Y_CTQ_ASSEMBLY,
    SMALL_Y_DEFECT_WASHING,
    SMALL_Y_CTQ_ACTIVATION
)

PROCESS_STEPS = [
    'Mixing',
    'Coating',
    'Roll Pressing',
    'Slitting',
    'Winding',
    'Assembly',
    'Activation'
]

DATA_TYPES = [
    'SV',
    'PV',
    'Small Y',
    'IQC'
]

INPUT_PROFILE = {
    # 'PV-Mixing' : [
    #     'SV-Mixing' 
    # ],
    # 'Small_Y-Mixing' : [
    #     'PV-Mixing',
    #     'SV-Mixing'
    # ],
    # 'PV-Coating' : [
    #     'PV-Mixing',
    # ],
    # 'Small_Y-Coating' : [
    #     'PV-Mixing',
    #     'PV-Coating'
    # ],
    # 'PV-Roll_Pressing' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'SV-Roll Pressing'
    # ],
    # # 'Small_Y_Roll_Pressing' : [
        
    # # ],
    # 'PV-Slitting' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'SV-Slitting'
    # ],
    # 'Small_Y-Slitting' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'SV-Slitting',
    #     'PV-Slitting'
    # ],
    # 'PV-Winding' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'PV-Slitting',
    #     'SV-Winding'
    # ],
    # 'Small_Y-Winding' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'PV-Slitting',
    #     'SV-Winding',
    #     'PV-Winding'
    # ],
    # 'PV-Assembly' : [
    #     'PV-Mixing',
    #     'PV-Coating',
    #     'PV-Slitting',
    #     'PV-Winding',
    #     'SV-Assembly'
    # ],
    # 'Small_Y_Assembly' : [

    # ],
    # 'PV_Activation' : [

    # ],
    # 'Small_Y_Activation' : [

    # ],
    'Big_Y-Activation' : [
        'PV-Mixing',
        'PV-Coating',
        'PV-Slitting',
        'PV-Winding',
        'PV-Assembly',
        'Small_Y-Coating',
        'Small_Y-Slitting',
        'Small_Y-Winding',
        #'SV-Activation'
    ],
    # 'Big_YSV-Activation' : [
    #     'SV-Mixing',
    #     'SV-Roll Pressing',
    #     'SV-Slitting',
    #     'SV-Winding',
    #     'SV-Assembly',
    #     'SV-Activation'
    # ]
    

}

def clean_col_name(col):
    col=str(col)
    col=re.sub(R"_+", "_",col)
    col=re.sub(r"[^가-힣a-zA-Z0-9_]", "_", col)
    return col

def squeeze_list(data_list):
    data_list= [x for sublist in data_list for x in sublist]
    return data_list

class ColumnClassifier:
    def __init__(self):
        pass
    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        # IQC
        list_cols_iqc_mixing = [
            x for x in data.columns
            if 'IQC' in x
            and 'Electrode' in x
        ]

        list_cols_iqc_assembly = [
            x for x in data.columns
            if 'IQC' in x
            and 'Assembly' in x
        ]

        # SV
        list_cols_sv_mixing = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Mixing' in x
            and 'DV' not in x
        ]

        list_cols_sv_coating = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Coating' in x
            and 'DV' not in x
        ]

        list_cols_sv_roll_pressing = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Roll Pressing' in x
            and 'DV' not in x
        ]

        list_cols_sv_slitting = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Slitting' in x
            and 'DV' not in x
        ]

        list_cols_sv_winding = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Winding' in x
            and 'DV' not in x
        ]

        list_cols_sv_assembly = [
            x for x in data.columns 
            if 'SV' in x 
            and 'Assembly' in x
            and 'DV' not in x
        ]

        list_cols_sv_activation = [
            # x for x in data.columns 
            # if 'NFF' in x 
            # and 'DV' not in x
            # and 'Y_NFF_A' not in x
            # and 'Y_NFF_B' not in x
            # and 'Y_NFF_C' not in x
            # and 'Y_NFF_D' not in x
            # and 'Y_NFF_E' not in x
            # and 'Y_NFF_F' not in x
            # and 'Y_NFF_G' not in x
            # and 'Y_NFF_H' not in x
            # and 'Y_NFF_I' not in x
            # and 'Y_NFF_J' not in x
            # and 'Y_NFF_K' not in x
            # and 'Y_NFF_L' not in x
            # and 'Y_NFF_O' not in x
            # and 'Y_NFF_P' not in x
            # and 'Y_NFF_Q' not in x
            # and 'Y_NFF_R' not in x
            # and 'Y_NFF_T' not in x
            # and 'Y_NFF_V' not in x
            # and 'Y_NFF_W' not in x
            # and 'Y_NFF_Z' not in x
        ]
        

        # PV
        list_cols_pv_mixing = [
            x for x in data.columns 
            if 'PV' in x 
            and 'Mixing' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_coating = [
            x for x in data.columns 
            if 'PV' in x 
            and 'Coating' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_roll_pressing = [
            x for x in data.columns 
            if 'PV' in x 
            and 'RollPress' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_slitting = [
            x for x in data.columns 
            if 'PV' in x 
            and 'Slitting' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_winding = [
            x for x in data.columns 
            if 'PV' in x 
            and 'Winding' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_assembly = [
            x for x in data.columns 
            if 'PV' in x 
            and 'Assembly' in x
            and 'SV' not in x
            and 'DV' not in x
        ]

        list_cols_pv_activation = [
            # x for x in data.columns 
            # if 'NFF' in x 
            # and 'SV' not in x
            # and 'DV' not in x
            # and 'Grade' not in x
        ]
        

        # Small Y
        list_cols_small_y_electrode = (
        SMALL_Y_DEFECT_ELECTRODE['검사불량이력']
        + SMALL_Y_DEFECT_ELECTRODE['생산이력데이터']
        + SMALL_Y_DEFECT_ELECTRODE['LQC']
        + SMALL_Y_DEFECT_ELECTRODE['SPC']

        + SMALL_Y_CTQ_ELECTRODE['공정자주샘플링검사']
        + SMALL_Y_CTQ_ELECTRODE['LQC']
        + SMALL_Y_CTQ_ELECTRODE['PV']
        )

        list_cols_small_y_mixing = [
            x for x in list_cols_small_y_electrode if 'Mixing' in x
        ]
        list_cols_small_y_mixing = list(set([x for x in list_cols_small_y_mixing if x in data.columns]))

        list_cols_small_y_coating = [
            x for x in list_cols_small_y_electrode if 'Coating' in x
        ]
        list_cols_small_y_coating = list(set([x for x in list_cols_small_y_coating if x in data.columns]))

        list_cols_small_y_roll_pressing = [
            x for x in list_cols_small_y_electrode if 'Roll Pressing' in x
        ]
        list_cols_small_y_roll_pressing = list(set([x for x in list_cols_small_y_roll_pressing if x in data.columns]))

        list_cols_small_y_slitting = [
            x for x in list_cols_small_y_electrode if 'Slitting' in x
        ]
        list_cols_small_y_slitting = list(set([x for x in list_cols_small_y_slitting if x in data.columns]))


        list_cols_small_y_winding = (
            SMALL_Y_DEFECT_WINDING['검사불량이력']
            + SMALL_Y_CTQ_WINDING['SPC']
            + SMALL_Y_CTQ_WINDING['검사불량이력']
            + SMALL_Y_CTQ_WINDING['PV']
        )
        list_cols_small_y_winding = list(set([x for x in list_cols_small_y_winding if x in data.columns]))

        list_cols_small_y_assembly = (
            SMALL_Y_DEFECT_ASSEMBLY['검사불량이력']
            + SMALL_Y_DEFECT_ASSEMBLY['cIoT']
            + SMALL_Y_DEFECT_ASSEMBLY['SPC']
            + SMALL_Y_CTQ_ASSEMBLY['LQC']
            + SMALL_Y_CTQ_ASSEMBLY['SPC']
        )
        list_cols_small_y_assembly = list(set([x for x in list_cols_small_y_assembly if x in data.columns]))

        list_cols_small_y_activation = (
            SMALL_Y_DEFECT_WASHING['SPC']
            + SMALL_Y_CTQ_ACTIVATION['Lot별 Cell Data']
            + SMALL_Y_CTQ_ACTIVATION['NFF']
        )
        list_cols_small_y_activation = list(set([x for x in list_cols_small_y_activation if x in data.columns]))

        # Big Y
        list_cols_big_y_activation = [
            'Y_NFF_A',
            'Y_NFF_D',
            'Y_NFF_E',
            'Y_NFF_F',
            'Y_NFF_H',
            'Y_NFF_J',
            'Y_NFF_K',
            'Y_NFF_L',
            'Y_NFF_O',
            'Y_NFF_P',
            'Y_NFF_Q',
            'Y_NFF_R',
            'Y_NFF_T',
            'Y_NFF_V',
            'Y_NFF_W',
            'Y_NFF_Z'
        ]

        df_cols = [
        # IQC
         {
            'process_step' : 'Mixing',
            'data_type' : 'IQC',
            'cols' : list_cols_iqc_mixing,
            'count' : len(list_cols_iqc_mixing)
        },
        {
            'process_step' : 'Assembly',
            'data_type' : 'IQC',
            'cols' : list_cols_iqc_assembly,
            'count' : len(list_cols_iqc_assembly)
        },
        # SV
         {
            'process_step' : 'Mixing',
            'data_type' : 'SV',
            'cols' : list_cols_sv_mixing,
            'count' : len(list_cols_sv_mixing)
        },
        {
            'process_step' : 'Coating',
            'data_type' : 'SV',
            'cols' : list_cols_sv_coating,
            'count' : len(list_cols_sv_coating)
        },
        {
            'process_step' : 'Roll Pressing',
            'data_type' : 'SV',
            'cols' : list_cols_sv_roll_pressing,
            'count' : len(list_cols_sv_roll_pressing)
        },
        {
            'process_step' : 'Slitting',
            'data_type' : 'SV',
            'cols' : list_cols_sv_slitting,
            'count' : len(list_cols_sv_slitting)
        },
        {
            'process_step' : 'Winding',
            'data_type' : 'SV',
            'cols' : list_cols_sv_winding,
            'count' : len(list_cols_sv_winding)
        },
        {
            'process_step' : 'Assembly',
            'data_type' : 'SV',
            'cols' : list_cols_sv_assembly,
            'count' : len(list_cols_sv_assembly)
        },
         {
            'process_step' : 'Activation',
            'data_type' : 'SV',
            'cols' : list_cols_sv_activation,
            'count' : len(list_cols_sv_activation)
        },
        # PV
         {
            'process_step' : 'Mixing',
            'data_type' : 'PV',
            'cols' : list_cols_pv_mixing,
            'count' : len(list_cols_pv_mixing)
        },
        {
            'process_step' : 'Coating',
            'data_type' : 'PV',
            'cols' : list_cols_pv_coating,
            'count' : len(list_cols_pv_coating)
        },
        {
            'process_step' : 'Roll Pressing',
            'data_type' : 'PV',
            'cols' : list_cols_pv_roll_pressing,
            'count' : len(list_cols_pv_roll_pressing)
        },
        {
            'process_step' : 'Slitting',
            'data_type' : 'PV',
            'cols' : list_cols_pv_slitting,
            'count' : len(list_cols_pv_slitting)
        },
        {
            'process_step' : 'Winding',
            'data_type' : 'PV',
            'cols' : list_cols_pv_winding,
            'count' : len(list_cols_pv_winding)
        },
        {
            'process_step' : 'Assembly',
            'data_type' : 'PV',
            'cols' : list_cols_pv_assembly,
            'count' : len(list_cols_pv_assembly)
        },
         {
            'process_step' : 'Activation',
            'data_type' : 'PV',
            'cols' : list_cols_pv_activation,
            'count' : len(list_cols_pv_activation)
        },
        # Small Y
        {
            'process_step' : 'Mixing',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_mixing,
            'count' : len(list_cols_small_y_mixing)
        },
        {
            'process_step' : 'Coating',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_coating,
            'count' : len(list_cols_small_y_coating)
        },
        {
            'process_step' : 'Roll Pressing',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_roll_pressing,
            'count' : len(list_cols_small_y_roll_pressing)
        },
        {
            'process_step' : 'Slitting',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_slitting,
            'count' : len(list_cols_small_y_slitting)
        },
        {
            'process_step' : 'Winding',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_winding,
            'count' : len(list_cols_small_y_winding)
        },
        {
            'process_step' : 'Assembly',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_assembly,
            'count' : len(list_cols_small_y_assembly)
        },
         {
            'process_step' : 'Activation',
            'data_type' : 'Small_Y',
            'cols' : list_cols_small_y_activation,
            'count' : len(list_cols_small_y_activation)
        },
        # Big Y
        {
            'process_step' : 'Activation',
            'data_type' : 'Big_Y',
            'cols' : list_cols_big_y_activation,
            'count' : len(list_cols_big_y_activation)
        }
        ]

        df_cols = pd.DataFrame(df_cols)


        return df_cols