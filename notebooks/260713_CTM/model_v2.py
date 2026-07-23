

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from IPython.display import display
import pandas as pd
import re

import seaborn as sns
import xgboost as xgb
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

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
from utils import squeeze_list

def clean_col_name(col):
    col=str(col)
    col=re.sub(R"_+", "_",col)
    col=re.sub(r"[^가-힣a-zA-Z0-9_]", "_", col)
    return col

class CascadeModel:
    def __init__(self, df_cols):
        self.pred = {

        }
        self._model_sv_iqc_to_big_y = None
        self._model_pv_to_small_y = None
        self._model_small_y_to_big_y = None

        self.df_cols = df_cols
        
    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame) -> None : 
        # Step 1 : SV + IQC -> PV

        ## 1-1. Input columns
        ### 1-1-1. SV
        cols_input_sv = (
            self.df_cols
            .loc[lambda x : x['data_type'] == 'SV']
            .loc[:, 'cols']
            .tolist()
        )
        cols_input_sv = squeeze_list(cols_input_sv)

        ### 1-1-2. IQC
        cols_input_iqc = (
            self.df_cols
            .loc[lambda x : x['data_type'] == 'IQC']
            .loc[:, 'cols']
            .tolist()
        )
        cols_input_iqc = squeeze_list(cols_input_iqc)

        cols_input = (
            cols_input_sv
            + cols_input_iqc
        )

        ## 1-2. Output columns
        cols_output = (
            self.df_cols
            .loc[lambda x : x['data_type'] == 'Big_Y']
            .loc[:, 'cols']
            .tolist()
        )
        cols_output = squeeze_list(cols_output)

        x_train = x_train.loc[:, cols_input]
        y_train = y_train.loc[:, cols_output]
        # print(f"x.shape : {x_train.shape}, y.shape : {y_train.shape}")
        # display(x_train)
        # display(y_train)

        x_train.columns = [clean_col_name(c) for c in x_train.columns]

        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            multi_strategy='multi_output_tree',
            # GPU parameters
            tree_method='hist',
            device='cuda:0',
            n_jobs=-1

        )

        reg.fit(x_train, y_train)
        self._model_sv_iqc_to_big_y = reg



        # Step 2 : (SV+IQC) + PV -> Small y

        # Step 3 : (SV+IQC+PV) + Small y -> Big Y

        return

    def predict(self, x_test:pd.DataFrame) -> pd.DataFrame:
        # Step 1 : SV + IQC -> PV

        ## 1-1. Input columns
        ### 1-1-1. SV
        cols_input_sv = (
            self.df_cols
            .loc[lambda x : x['data_type'] == 'SV']
            #.loc[lambda x : x['process_step']==input_process_step]
            .loc[:, 'cols']
            .tolist()
        )
        cols_input_sv = squeeze_list(cols_input_sv)

        ### 1-1-2. IQC
        cols_input_iqc = (
            self.df_cols
            .loc[lambda x : x['data_type'] == 'IQC']
            #.loc[lambda x : x['process_step']==input_process_step]
            .loc[:, 'cols']
            .tolist()
        )
        cols_input_iqc = squeeze_list(cols_input_iqc)

        cols_input = (
            cols_input_sv
            + cols_input_iqc
        )

        x_test = x_test.loc[:, cols_input]
        #display(x_test)

        x_test.columns = [clean_col_name(c) for c in x_test.columns]

        reg = self._model_sv_iqc_to_big_y
        pred = reg.predict(x_test)
        return pred
