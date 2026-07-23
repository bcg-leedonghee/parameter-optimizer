

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
from typing import List

def clean_col_name(col):
    col=str(col)
    col=re.sub(R"_+", "_",col)
    col=re.sub(r"[^가-힣a-zA-Z0-9_]", "_", col)
    return col

class CascadeModel:
    def __init__(self, cols_small_y : List, cols_big_y : List):
        self._reg1 = xgb.XGBRegressor(
            n_estimators=300,
            #max_depth=6,
            max_bin=64,
            learning_rate=0.05,
            #max_cached_hist_node=2048,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            #grow_policy='lossguide',
            random_state=42,
            multi_strategy='one_output_per_tree',
            # GPU parameters
            tree_method='hist',
            device='cuda:0',
            n_jobs=-1

        )
        self._reg2 = xgb.XGBRegressor(
            n_estimators=300,
            #max_depth=6,
            max_bin=64,
            learning_rate=0.05,
            #max_cached_hist_node=2048,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            #grow_policy='lossguide',
            random_state=42,
            multi_strategy='one_output_per_tree',
            # GPU parameters
            tree_method='hist',
            device='cuda:0',
            n_jobs=-1
        )
        self.cols_small_y = cols_small_y
        self.cols_big_y = cols_big_y

    def fit(self, x_train, y_train) :
        


        # 1단계 모델 학습
        # Input : SV + IQC
        # Output : Small y
        x_train_1 = x_train
        x_train_1 = x_train_1.loc[:, ~x_train_1.columns.duplicated()]

        y_train_1 = y_train.loc[:, self.cols_small_y]
        y_train_1 = y_train_1.loc[:, ~y_train_1.columns.duplicated()]
        self.y_train_1 = y_train_1

        x_train_1.columns = [clean_col_name(x) for x in x_train_1.columns]
        y_train_1.columns = [clean_col_name(x) for x in y_train_1.columns]
        self._reg1.fit(x_train_1, y_train_1)

        # 2단계 모델 학습
        # Input : SV + IQC + Small y
        # Output : Big Y
        x_train_2 = pd.concat([x_train, y_train.loc[:, self.cols_small_y]], axis=1)
        x_train_2 = x_train_2.loc[:, ~x_train_2.columns.duplicated()]

        y_train_2 = y_train.loc[:, self.cols_big_y]
        y_train_2 = y_train_2.loc[:, ~y_train_2.columns.duplicated()]
        self.y_train_2 = y_train_2

        x_train_2.columns = [clean_col_name(x) for x in x_train_2.columns]
        y_train_2.columns = [clean_col_name(x) for x in y_train_2.columns]
        self._reg2.fit(x_train_2, y_train_2)
    
    def predict(self, x_test):
        print(x_test.shape)
        # 1단계 모델 예측
        # Input : SV + IQC
        # Output : Small y
        x_test_1 = x_test

        x_test_1 = x_test_1.loc[:, ~x_test_1.columns.duplicated()]


        x_test_1.columns = [clean_col_name(x) for x in x_test_1.columns]


        y_pred_1 = self._reg1.predict(x_test_1)

        y_pred_1 = pd.DataFrame(y_pred_1)

        y_pred_1.columns = self.y_train_1.columns
        y_pred_1 = y_pred_1.loc[:, ~y_pred_1.columns.duplicated()]

        
        # 2단계 모델 예측
        # Input : SV + IQC + Small y (predicted)
        # Output : Big Y

        x_test_2 = pd.concat([x_test_1.reset_index(drop=True), y_pred_1.reset_index(drop=True)], axis=1)

        x_test_2 = x_test_2.loc[:, ~x_test_2.columns.duplicated()]


        #x_test_2.columns = [clean_col_name(x) for x in x_test_2.columns]


        y_pred_2 = self._reg2.predict(x_test_2)

        y_pred_2 = pd.DataFrame(y_pred_2)

        y_pred_2.columns = self.y_train_2.columns


        pred = {
            'big_y' : y_pred_2,
            'small_y' : y_pred_1
        }

        return pred
