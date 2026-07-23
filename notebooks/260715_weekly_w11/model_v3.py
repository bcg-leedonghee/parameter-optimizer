

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
    def __init__(self):
        self._reg = xgb.XGBRegressor(
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

    def fit(self, x_train, y_train) :
        self._reg.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self._reg.predict(x_test)
