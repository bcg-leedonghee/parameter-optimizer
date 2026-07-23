

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
        # self._reg2 = xgb.XGBRegressor(
        #     n_estimators=300,
        #     #max_depth=6,
        #     max_bin=64,
        #     learning_rate=0.05,
        #     #max_cached_hist_node=2048,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     objective='reg:squarederror',
        #     #grow_policy='lossguide',
        #     random_state=42,
        #     multi_strategy='one_output_per_tree',
        #     # GPU parameters
        #     tree_method='hist',
        #     device='cuda:0',
        #     n_jobs=-1
        # )
        self._clf2 = xgb.XGBClassifier(
            n_estimators=300,
            #max_depth=6,
            max_bin=64,
            learning_rate=0.05,
            #max_cached_hist_node=2048,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric="logloss",
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
        # Output : Small y (공정 중간 품질 지표, 예: 전극/권취/조립/세척 공정의 불량 및 CTQ)
        
        # 1단계 입력 데이터: 원본 학습 데이터(SV + IQC)를 그대로 사용
        x_train_1 = x_train
        
        # 동일한 이름의 컬럼이 중복으로 존재할 경우 첫 번째 컬럼만 유지 (중복 컬럼 제거)
        x_train_1 = x_train_1.loc[:, ~x_train_1.columns.duplicated()]

        # 1단계 출력 데이터: y_train에서 Small y에 해당하는 컬럼만 선택
        y_train_1 = y_train.loc[:, self.cols_small_y]
        
        # Small y 컬럼 중에도 중복이 있을 수 있으므로 동일하게 중복 제거
        y_train_1 = y_train_1.loc[:, ~y_train_1.columns.duplicated()]
        
        # predict() 단계에서 예측 결과의 컬럼명 복원에 사용하기 위해 인스턴스 변수로 저장
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
        self._clf2.fit(x_train_2, y_train_2)
    
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


        y_pred_2 = self._clf2.predict(x_test_2)
        y_pred_2 = pd.DataFrame(y_pred_2)
        y_pred_2.columns = self.y_train_2.columns

        y_pred_2_proba = self._clf2.predict_proba(x_test_2)
        y_pred_2_proba = pd.DataFrame(y_pred_2_proba)
        y_pred_2_proba.columns = self.y_train_2.columns


        pred = {
            'big_y' : y_pred_2,
            'big_y_proba' : y_pred_2_proba,
            'small_y' : y_pred_1
        }

        return pred
