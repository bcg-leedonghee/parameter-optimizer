

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

def clean_col_name(col):
    col=str(col)
    col=re.sub(R"_+", "_",col)
    col=re.sub(r"[^가-힣a-zA-Z0-9_]", "_", col)
    return col

class CascadeModel:
    def __init__(self, input_profile, df_cols):
        self.pred = {

        }
        self._model = None
        self.input_profile = input_profile
        self.df_cols = df_cols
    def fit(self, x_train:pd.DataFrame, y_train:pd.DataFrame) -> None : 
        for input_profile in self.input_profile.keys() :
            #print(input_profile)
            data_type, process_step = input_profile.split('-')
            #print(data_type, process_step)

            target_profile_str = f"{data_type}-{process_step}"

            input_profiles_str = self.input_profile[target_profile_str]
            #print(f"target : {target_profile_str}, input : {input_profiles_str}")

            cols_output = (
                self.df_cols
                .loc[lambda x : x['data_type'] == data_type]
                .loc[lambda x : x['process_step']==process_step]
                .loc[:, 'cols']
            )
            cols_output = [x for sublist in cols_output for x in sublist]
            #print(f"cols_output : {cols_output[:5]}")

            cols_input = []
            for input_profile_str in input_profiles_str : 
                input_data_type, input_process_step = input_profile_str.split('-')
                cols_input_chunk = (
                    self.df_cols
                    .loc[lambda x : x['data_type'] == input_data_type]
                    .loc[lambda x : x['process_step']==input_process_step]
                    .loc[:, 'cols']
                )
                cols_input_chunk = [x for sublist in cols_input_chunk for x in sublist]
                cols_input += cols_input_chunk
                #print(f"input data type : {input_data_type}, input process step : {input_process_step}")
            #print(f"cols_input : {cols_input[:5]}")

            x_train = x_train.loc[:, cols_input]
            y_train = y_train.loc[:, cols_output]
            #print(f"x.shape : {x.shape}, y.shape : {y.shape}")

            x_train.columns = [clean_col_name(c) for c in x_train.columns]

            #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            reg = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42,
                multi_strategy='multi_output_tree'
            )

            reg.fit(x_train, y_train)
            self._model = reg

        return

    def predict(self, x_test:pd.DataFrame) -> pd.DataFrame:
        for input_profile in tqdm(self.input_profile.keys()) :
            #print(f"preidct : {input_profile}")
            data_type, process_step = input_profile.split('-')
            #print(f"predict : {data_type}, {process_step}")

            target_profile_str = f"{data_type}-{process_step}"
            input_profiles_str = self.input_profile[target_profile_str]
            #print(f"target : {target_profile_str}, input : {input_profiles_str}")
        

            cols_input = []
            for input_profile_str in input_profiles_str : 

                input_data_type, input_process_step = input_profile_str.split('-')
                cols_input_chunk = (
                    self.df_cols
                    .loc[lambda x : x['data_type'] == input_data_type]
                    .loc[lambda x : x['process_step']==input_process_step]
                    .loc[:, 'cols']
                )
                cols_input_chunk = [x for sublist in cols_input_chunk for x in sublist]
                cols_input += cols_input_chunk
                print(f"input data type : {input_data_type}, input process step : {input_process_step}")

            x_test = x_test.loc[:, cols_input]

            x_test.columns = [clean_col_name(c) for c in x_test.columns]



            reg = self._model
            y_pred = reg.predict(x_test)

            return y_pred