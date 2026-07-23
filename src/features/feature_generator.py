# isort:imports-stdlib
from __future__ import annotations
from typing import Dict, List, Optional

# isort:imports-thirdparty
import numpy as np
import pandas as pd

# isort:imports-firstparty

# isort:import-localfolder

# 조립 주액 압력
COLS_ASSEMBLY_PRESSURE = {
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0106": "주액 챔버 진공펌프 압력(Kpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0107": "주액 챔버 진공펌프 압력(Kpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0108": "주액 챔버 서보밸브 압력(Kpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0109": "주액 챔버 서보밸브 압력(Kpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0111": "함침 챔버 O-RING 진공 압력(Kpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0112": "함침 챔버 O-RING 진공 압력(Kpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0117": "함침 챔버 STEP#1 공정 압력(Kpa&Mpa",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0118": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0119": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0130": "함침 챔버 STEP#2 공정 압력(Kpa&Mpa",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0131": "함침 챔버 STEP#2 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0143": "함침 챔버 STEP#3 공정 압력(Kpa&Mpa",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0144": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0145": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0156": "함침 챔버 STEP#4 공정 압력(Kpa&Mpa",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0157": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0158": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0169": "함침 챔버 STEP#5 공정 압력(Kpa&Mpa",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0170": "함침 챔버 STEP#5 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assemble_Assembly_ELF_END_fdcelf0171": "함침 챔버 STEP#5 공정 서보밸브 압력(Kpa&Mpa) End",
}

COLS_ASSEMBLY_INJECTION_TIME = {
    "X_PV_Assemble_Assembly_fdcelf0166": "함침 챔버 STEP#5 가압&진공 상승 시간(s",
    "X_PV_Assemble_Assembly_fdcelf0167": "함침 챔버 STEP#5 가압&진공 유지 시간(s",
    "X_PV_Assemble_Assembly_fdcelf0168": "함침 챔버 STEP#5 가압&진공 하강 시간(s",
}

COLS_WINDING_TENSION = {
    "X_PV_Assemble_Winding_WINDING_fdcwnd5001": "분리막(상) 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5002": "분리막(상) 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5004": "분리막(상) 1단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5005": "분리막(상) 1단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5007": "분리막(상) 2단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5008": "분리막(상) 2단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5009": "분리막(상) 3단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5010": "분리막(상) 3단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5011": "분리막(상) 터렛 회전 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5012": "분리막(상) 권취 감속시점 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5013": "분리막(상) 커팅 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5014": "분리막(상) 커팅 후 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5015": "분리막(하) 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5016": "분리막(하) 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5018": "분리막(하) 1단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5019": "분리막(하) 1단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5021": "분리막(하) 2단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5022": "분리막(하) 2단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5023": "분리막(하) 3단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5024": "분리막(하) 3단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5025": "분리막(하) 터렛 회전 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5026": "분리막(하) 권취 감속시점 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5027": "분리막(하) 커팅 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5028": "분리막(하) 커팅 후 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5029": "양극 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5030": "양극 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5032": "양극 1단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5033": "양극 1단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5035": "양극 2단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5036": "양극 2단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5037": "양극 3단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5038": "양극 3단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5039": "양극 권취 감속위치 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5040": "양극 커팅 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5041": "양극 커팅 후 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5042": "음극 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5043": "음극 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5045": "음극 1단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5046": "음극 1단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5048": "음극 2단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5049": "음극 2단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5050": "음극 3단 시작 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5051": "음극 3단 완료 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5052": "음극 권취 감속위치 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5053": "음극 커팅 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5054": "음극 커팅 후 투입 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5055": "양극 Buffer 자동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5056": "양극 Buffer 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5057": "음극 Buffer 자동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5058": "음극 Buffer 수동 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5059": "양극 스풀 장력",
    "X_PV_Assemble_Winding_WINDING_fdcwnd5060": "양극 스풀 장력"
}

COLS_WINDING_SPEED = {
    'X_PV_Assemble_Winding_WINDING_fdcwnd5460': "P1 1단 권취 속도",
    'X_PV_Assemble_Winding_WINDING_fdcwnd5463': "P1 2단 권취 속도",
    'X_PV_Assemble_Winding_WINDING_fdcwnd5465': "P1 3단 권취 속도",
    'X_PV_Assemble_Winding_WINDING_fdcwnd5466': "P1 분리막 잔량 권취 속도"

}

COLS_WASHING_TEMPERATURE = [
#   'X_SV_Assemble_Washing_Parameter Spec LSL_2차 건조 핫부스터#1 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec LSL_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Assemble_Washing_Parameter Spec LSL_2차 저장탱크 히터 SV값',
#   'X_SV_Assemble_Washing_Parameter Spec LSL_3차 건조 핫부스터#3 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec LSL_3차 건조 핫부스터#4 온도 SV값',
  'X_SV_Assemble_Washing_Parameter Spec LSL_순간온수기 온도 설정값',
#   'X_SV_Assemble_Washing_Parameter Spec USL_2차 건조 핫부스터#1 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec USL_2차 건조 핫부스터#2 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec USL_3차 건조 핫부스터#3 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec USL_3차 건조 핫부스터#4 온도 SV값',
#  'X_SV_Assemble_Washing_Parameter Spec USL_순간온수기 온도 설정값',
 'X_SV_Assemble_Washing_Parameter Value_2차 저장탱크 히터 SV값',
 'X_SV_Assemble_Washing_Paramter Target Value_1차 저장탱크 히터 SV값',
  #'X_SV_Assemble_Washing_Paramter Target Value_2차 건조 핫부스터#1 온도 SV값',
 #'X_SV_Assemble_Washing_Paramter Target Value_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Assemble_Washing_Paramter Target Value_2차 저장탱크 히터 SV값',
  #'X_SV_Assemble_Washing_Paramter Target Value_3차 건조 핫부스터#3 온도 SV값',
 #'X_SV_Assemble_Washing_Paramter Target Value_3차 건조 핫부스터#4 온도 SV값',
#   'X_SV_Assemble_Washing_Paramter Target Value_순간온수기 온도 설정값',
]

COLS_ELECTRODE_LOT_ID = [
    '02_Coating(Back)_Lot ID',
    #'03_Roll Pressing_Lot ID',
    #'04_Slitting_Lot ID'
]

COLS_WINDING_AUTO_CORRECTION = [
    'X_SV_Winding_Parameter Value_양극 파이널 EPC 외곽 보정률',
    'X_SV_Winding_Parameter Value_양극 파이널 EPC 코어 보정률',
     'X_SV_Winding_Parameter Value_음극 파이널 EPC 외곽 보정률',
    'X_SV_Winding_Parameter Value_음극 파이널 EPC 코어 보정률',
    'X_SV_Winding_Paramter Target Value_양극 파이널 EPC 외곽 보정률',
    'X_SV_Winding_Paramter Target Value_양극 파이널 EPC 코어 보정률',
    'X_SV_Winding_Paramter Target Value_음극 파이널 EPC 외곽 보정률',
    'X_SV_Winding_Paramter Target Value_음극 파이널 EPC 코어 보정률'
]

COLS_COATING_DRY_TEMPERATURE = {
    # 'X_PV_Electrode_Slitting_Cathode_W002200_N_FC_LocalSimple_mean1_tauresra': "Dryer [In] Lamp Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002201_N_FC_LocalSimple_mean1_tauresra' : "Dryer [In] Dry Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002202_N_FC_LocalSimple_mean1_tauresra' : "Dryer [In] Dry Temperature (Back) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002203_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Mid] Lamp Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002204_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Mid] Dry Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002205_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Mid] Dry Temperature (Back) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002206_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Out] Lamp Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002207_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Out] Dry Temperature (Top) PV",
    # 'X_PV_Electrode_Slitting_Cathode_W002208_N_FC_LocalSimple_mean1_tauresra' : "Dryer [Out] Dry Temperature (Back) PV",
    'X_PV_Electrode_Slitting_Anode_W002200_N_FC_LocalSimple_mean1_tauresrat': "Dryer [In] Lamp Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002201_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [In] Dry Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002202_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [In] Dry Temperature (Back) PV",
    'X_PV_Electrode_Slitting_Anode_W002203_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [Mid] Lamp Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002204_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [Mid] Dry Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002205_N_DN_Mean': "Dryer [Mid] Dry Temperature (Back) PV",
    'X_PV_Electrode_Slitting_Anode_W002206_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [Out] Lamp Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002207_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [Out] Dry Temperature (Top) PV",
    'X_PV_Electrode_Slitting_Anode_W002208_N_FC_LocalSimple_mean1_tauresrat' : "Dryer [Out] Dry Temperature (Back) PV",
}

COLS_COATING_FAN = {
    #'X_PV_Electrode_Slitting_Anode_W002211_N': "Dryer Exhaust Fan Output (In)",
    'X_PV_Electrode_Slitting_Cathode_W002211_N': "Dryer Exhaust Fan Output (Out)",
}

COLS_LIINE_SPEED = {
    'X_PV_Electrode_Slitting_Cathode_W001520_N' : "Thickness Average (DS)",
    'X_PV_Electrode_Slitting_Cathode_W001522_N' : "Thickness Average (OS)",
}

COLS_ROLL_PRESSING_THICKNESS = {
    'X_PV_Electrode_Roll Pressing_Anode_W004E30_N_DN_Mean' : "1st Thickness Average 1Lane",
    'X_PV_Electrode_Roll Pressing_Anode_W004E32_N_DN_Mean' : "1st Thickness Average 3Lane",
     'X_PV_Electrode_Roll Pressing_Anode_W004E70_N_FC_LocalSimple_mean1_tauresrat' : "2nd Thickness Average 1Lane",
     'X_PV_Electrode_Roll Pressing_Anode_W004E71_N_FC_LocalSimple_mean1_tauresrat' : "2nd Thickness Average 2Lane",
     'X_PV_Electrode_Roll Pressing_Anode_W004E72_N_FC_LocalSimple_mean1_tauresrat' : "2nd Thickness Average 3Lane",
     'X_PV_Electrode_Roll Pressing_Anode_W004E73_N_FC_LocalSimple_mean1_tauresrat' : "2nd Thickness Average 4Lane",
     
}

#hs 추가
COLS_ASSEMBLY_ELF_AMOUNT = {
    'X_PV_Assemble_Assembly_ELF_END_fdcelf0008' : '최종 토출량(g)'
    }

#hs 추가
COLS_ELECTRODE_ANODE_WIDTH = [
    'X_RollMap_Anode_THICK_AVG_2', 
    'X_RollMap_Anode_THICK_AVG_3',
    'X_RollMap_Anode_THICK_AVG_4',
    'Y_Anode_GJJJ_Roll Pressing_[1차]두께-1 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-1 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-1 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-2 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-2 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-2 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-3 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-3 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-3 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-4 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-4 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-4 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-5 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-5 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-5 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-6 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-6 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-6 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-7 Lane-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-7 Lane-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-7 Lane-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-우측_Side-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-우측_Side-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-우측_Side-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-1 Whopper-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-1 Whopper-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-1 Whopper-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-2 Whopper-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-2 Whopper-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-2 Whopper-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-3 Whopper-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-3 Whopper-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-3 Whopper-3 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-4 Whopper-1 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-4 Whopper-2 Point_Inspection Measure Value',
 'Y_Anode_GJJJ_Roll Pressing_두께-4 Whopper-3 Point_Inspection Measure Value'
]
#hs 추가
COLS_ELECTRODE_CATHOD_WIDTH = [
    'X_RollMap_Cathode_THICK_AVG_2', 'X_RollMap_Cathode_THICK_AVG_3'
    'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-1 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-1 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-1 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-2 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-2 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-2 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-3 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-3 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-3 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-4 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-4 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-4 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-5 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-5 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-5 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-6 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-6 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-6 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-7 Lane-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-7 Lane-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-7 Lane-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-우측_Side-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-우측_Side-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-우측_Side-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_[1차]두께-좌측_Side-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-1 Whopper-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-1 Whopper-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-1 Whopper-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-2 Whopper-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-2 Whopper-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-2 Whopper-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-3 Whopper-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-3 Whopper-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-3 Whopper-3 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-4 Whopper-1 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-4 Whopper-2 Point_Inspection Measure Value',
 'Y_Cathode_GJJJ_Roll Pressing_두께-4 Whopper-3 Point_Inspection Measure Value'
]

COLS_WINDING_CORE = {'X_PV_Assemble_Winding_WINDING_fdcwnd5496' : '권심 사이즈'}
COLS_WINDING_ACC = {'X_PV_Assemble_Winding_WINDING_fdcwnd5466' : 'P1 최종 권취 감속도',
                    'X_PV_Assemble_Winding_WINDING_fdcwnd5461' : 'P1 권취 가속도',
                    'X_PV_Assemble_Winding_WINDING_fdcwnd5461' : 'P1 권취 가속도'
                    }

def is_constant_or_all_na(s) :
    non_na = s.dropna()
    if len(non_na) == 0 :
        return True

    values = non_na.map(repr)
    return values.nunique() == 1

def filter_dv_cols(data) : 
    cols_dv = [x for x in data.columns if 'DV' in x]
    cols_not_dv = [x for x in data.columns if 'DV' not in x]

    mask = ~data.loc[:, cols_dv].apply(is_constant_or_all_na)
    mask = cols_not_dv + data.loc[:, cols_dv].loc[: , mask].columns.tolist()
    data = data.loc[:, mask]
    return data

class FeatureGenerator:
    def __init__(self):
        pass

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:

        # 원본 데이터 복사
        data_transformed = data.copy()

        # DV02 : 주액 압력 / 와인더 텐션
        # 260714 1255 동작 확인 완료
        
        data_transformed = (
            self._dv_02(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV02 : 주액 압력 / 와인더 텐션')

        # DV03 : 주액 압력 * 주액 시간 / 와인더 텐션
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_03(data_transformed)
        )
        print('# DV03 : 주액 압력 * 주액 시간 / 와인더 텐션')

        # DV06 : 주액 압력 *  워싱 온도 / 와인더 텐션
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_06(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV06 : 주액 압력 *  워싱 온도 / 와인더 텐션')

        # DV07 : 주액 압력 * 주액 시간 x 워싱 온도 / 와인더 텐션
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_07(data_transformed)
        )
        print('DV07 : 주액 압력 * 주액 시간 x 워싱 온도 / 와인더 텐션')

        # DV15 : 활성화 공정 종료 시간 - 조립 공정 종료 시간
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_15(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV15 : 활성화 공정 종료 시간 - 조립 공정 종료 시간')

        # # DV11 : 워싱 온도 / 와인더 텐션
        # # 260714 1255 확인 완료 -> 데이터 없음
        # data_transformed = (
        #     self._dv_11(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV11 : 워싱 온도 / 와인더 텐션')

        # DV13 : 전극 Lot 전환 플래그
        # 제외
        # data_transformed = (
        #     self._dv_13(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV13 : 전극 Lot 전환 플래그')

        # # DV16 : 슬리팅 이물 플래그
        # # 260714 1310 확인 완료 -> 데이터 없음
        # data_transformed = (
        #     self._dv_16(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV16 : 슬리팅 이물 플래그')

        # DV17 : 양극 음극 두께 잔차
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_17(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV17 : 양극 음극 두께 잔차')

        # DV20 : 와인딩 속도 x 텐션 tradeoff
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_20(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV20 : 와인딩 속도 x 텐션 tradeoff')

        # DV27 : 누적 성형 카운트 square / log
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_27(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV27 : 누적 성형 카운트 square / log')

        # DV29 : 주해액 Lot 대비 Deviation
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_29(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV29 : 주해액 Lot 대비 Deviation')

        # DV30: SEI 형성량
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_30(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV30: SEI 형성량')

        # DV35: 코팅 드라이
        # 260714 1310 확인 완료 -> 데이터 없음
        # data_transformed = (
        #     self._dv_35(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV35: 코팅 드라이')

        # DV36: 슬리팅/롤프레싱간 두께 차이
        # 260714 1320 확인 완료 -> 데이터 없음
        # data_transformed = (
        #     self._dv_36(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV36: 슬리팅/롤프레싱간 두께 차이')

        # DV40: 활성화 종료 시간 - 믹싱 종료 시간
        # 260714 1255 -> 오류
        data_transformed = (
            self._dv_40(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV40: 활성화 종료 시간 - 믹싱 종료 시간')

        # DV42: 미세라인 (점도 * 고형분) + (코팅갭 * 슬러리 코팅온도)
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_42(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV42: 미세라인 (점도 * 고형분) + (코팅갭 * 슬러리 코팅온도)')

        # DV47: 전극주름 max텐션 - min텐션
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_47(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV47: 전극주름 max텐션 - min텐션')
            
        # DV52: (양극 + 음극 + 분리막 두께) x 젤리롤 턴수
        # 260714 1255 동작 확인 완료
        data_transformed = (
            self._dv_52(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV52: (양극 + 음극 + 분리막 두께) x 젤리롤 턴수')

        # DV56: (양극 + 음극 + 분리막 두께) x 젤리롤 턴수
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_56(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV56: (양극 + 음극 + 분리막 두께) x 젤리롤 턴수')

        # DV83: 양극 로딩량 * 음극 로딩량
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_83(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV83: 양극 로딩량 * 음극 로딩량')

        # DV84: 점도 * 라인 속도 --> 관계 없다는 피드백 받음. 제외
        # data_transformed = (
        #     self._dv_84(data_transformed)
        # )
        # data_transforemd = filter_dv_cols(data_transformed)
        # print('# DV84: 점도 * 라인 속도')

        # DV85: 와인더 감속 * 장력
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_85(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV85: 와인더 감속 * 장력')

        # DV86: 와인더 가속 * 장력
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_86(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV86: 와인더 가속 * 장력')

        # DV88: 
        # 260714 1310 동작 확인 완료
        data_transformed = (
            self._dv_88(data_transformed)
        )
        data_transforemd = filter_dv_cols(data_transformed)
        print('# DV88: ')




        return data_transformed

    def _dv_02(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV02_'
        data_transformed = data
        results = []

        for col_pressure, desc_pressure in zip(COLS_ASSEMBLY_PRESSURE.keys(), COLS_ASSEMBLY_PRESSURE.values()) :
            for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                try : 
                    # data_transformed[f"{prefix}_{desc_pressure}/{desc_tension}"] = (
                    #     data[col_pressure] / data[col_tension]
                    # )
                    ratio = (
                        data[col_pressure] / data[col_tension]
                    )
                    results.append((f"{prefix}_{desc_pressure}/{desc_tension}", ratio))
                except : 
                    pass

        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_03(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV03_'
        data_transformed = data
        results = []

        for col_pressure, desc_pressure in zip(COLS_ASSEMBLY_PRESSURE.keys(), COLS_ASSEMBLY_PRESSURE.values()) :
            for col_time, desc_time in zip(COLS_ASSEMBLY_INJECTION_TIME.keys(), COLS_ASSEMBLY_INJECTION_TIME.values()) : 
                for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                    try : 
                        # data_transformed[f"{prefix}_{desc_pressure}x_{desc_time}/{desc_tension}"] = (
                        #     data[col_pressure] * data[col_time] / data[col_tension]
                        # )
                        ratio = (
                            data[col_pressure] * data[col_time] / data[col_tension]
                        )
                        results.append((f"{prefix}_{desc_pressure}x_{desc_time}/{desc_tension}", ratio))
                    except : 
                        pass

        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_06(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV06_'
        data_transformed = data
        results = []

        for col_pressure, desc_pressure in zip(COLS_ASSEMBLY_PRESSURE.keys(), COLS_ASSEMBLY_PRESSURE.values()) :
            for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                for col_washing_temp in COLS_WASHING_TEMPERATURE : 
                    #try : 
                    data_transformed[f"{prefix}_{desc_pressure}x_{col_washing_temp}/{desc_tension}"] = (
                        data[col_pressure] * data[col_washing_temp] / data[col_tension]
                    )
                        #     ratio = (
                        #         data[col_pressure] * data[col_washing_temp] / data[col_tension]
                        #     )
                        #     results.append((f"{prefix}_{desc_pressure}x_{col_washing_temp}/{desc_tension}", ratio))
                    # except Exception as e: 
                    #     print(e)
        # 한 번에 병합
        # if results:
        #     result_df = pd.DataFrame({k: v for k, v in results})
        #     data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    
    def _dv_07(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV07_'
        data_transformed = data
        results = []


        for col_pressure, desc_pressure in zip(COLS_ASSEMBLY_PRESSURE.keys(), COLS_ASSEMBLY_PRESSURE.values()) :
            for col_time, desc_time in zip(COLS_ASSEMBLY_INJECTION_TIME.keys(), COLS_ASSEMBLY_INJECTION_TIME.values()) : 
                for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                    for col_washing_temp in COLS_WASHING_TEMPERATURE : 
                        try : 
                            data_transformed[f"{prefix}_{desc_pressure}x_{desc_time}x_{col_washing_temp}/{desc_tension}"] = (
                                data[col_pressure] * data[col_time] * data[col_washing_temp] / data[col_tension]
                            )
                            ratio = (
                                data[col_pressure] * data[col_time] * data[col_washing_temp] / data[col_tension]
                            )
                            results.append((f"{prefix}_{desc_pressure}x_{desc_time}x_{col_washing_temp}/{desc_tension}", ratio))
                        except Exception as e: 
                            pass
                           #print(e)
        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_15(self, data:pd.DataFrmae) -> pd.DataFrame:
        prefix = 'DV15_'
        data_transformed = data

        col_start = '06_Assembly_Finished Date'
        col_end = '07_Before Degas_Finished Date'

        def tmp(x):
            try : 
                x = x[0]
            except : 
                pass
            return x

        data_transformed[col_start] = pd.to_datetime(data_transformed[col_start].apply(lambda x : tmp(x)), errors='coerce')
        data_transformed[col_end] = pd.to_datetime(data_transformed[col_end].apply(lambda x : tmp(x)), errors='coerce')

        data_transformed[f"{prefix}_{col_end}-{col_start}"] = ((
            data_transformed[col_end] - data_transformed[col_start]
        ).dt.total_seconds()//3600)
        return data_transformed

    def _dv_11(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV07_'
        data_transformed = data
        results = []


        for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
            for col_washing_temp in COLS_WASHING_TEMPERATURE : 
                try : 
                    # data_transformed[f"{prefix}_{col_washing_temp}/{desc_tension}"] = (
                    #     data[col_washing_temp] / data[col_tension]
                    # )
                    ratio = (
                        data[col_washing_temp] / data[col_tension]
                    )
                    results.append((f"{prefix}_{col_washing_temp}/{desc_tension}", ratio))
                except Exception as e: 
                    pass
                    #print(e)
        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_13(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV13_'
        data_transformed = data

        # DV13__전극 LOT 전환 플래그
        for col_lot in COLS_ELECTRODE_LOT_ID : 
            electrode_lot = data_transformed[col_lot]
            lot_change_flag = (electrode_lot != electrode_lot.shift(1)).astype("Int8")
            n_cells = 50
            data_transformed[f"{prefix}_{electrode_lot}_change"] = lot_change_flag.rolling(window=n_cells, min_periods=1).max()
        return data_transformed

    def _dv_16(self, data: pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV16_'
        data_transformed = data
        
        slit_speed_anode = {
            "X_PV_Electrode_Slitting_Anode_D0006204_N" : "Line Speed SV"
        }
        slit_speed_cathode = {
            "X_PV_Electrode_Slitting_Cathode_D0006204_N" : "Line Speed SV"
        }
        slit_tension_anode = {
            'X_PV_Electrode_Slitting_Anode_R00155_N' : "Unwinder Tension SV",
            "X_PV_Electrode_Slitting_Anode_R00158_N": "TM Outfeed Tension SV",
            "X_PV_Electrode_Slitting_Anode_R00159_N": "Rewinder Upper Tension Taper",
            "X_PV_Electrode_Slitting_Anode_R00160_N": "Rewinder Lower Tension Taper",
            "X_PV_Electrode_Slitting_Anode_R00161_N": "Rewinder Upper Tension SV",
            "X_PV_Electrode_Slitting_Anode_R00162_N": "Rewinder Lower Tension SV",
        }
        slit_tension_cathode = {
            'X_PV_Electrode_Slitting_Cathode_R00155_N' : "Unwinder Tension SV",
            "X_PV_Electrode_Slitting_Cathode_R00158_N": "TM Outfeed Tension SV",
            "X_PV_Electrode_Slitting_Cathode_R00159_N": "Rewinder Upper Tension Taper",
            "X_PV_Electrode_Slitting_Cathode_R00160_N": "Rewinder Lower Tension Taper",
            "X_PV_Electrode_Slitting_Cathode_R00161_N": "Rewinder Upper Tension SV",
            "X_PV_Electrode_Slitting_Cathode_R00162_N": "Rewinder Lower Tension SV",
        }

        for col_speed, desc_speed in zip(slit_speed_anode.keys(), slit_speed_anode.values()) : 
            for col_tension, desc_tension in zip(slit_tension_anode.keys(), slit_tension_anode.values()) :
                data_transformed[f"{prefix}_Anode_{desc_speed}x{desc_tension}"] = (
                    data_transformed[col_speed] * data_transformed[col_tension]
                )

        for col_speed, desc_speed in zip(slit_speed_cathode.keys(), slit_speed_cathode.values()) : 
            for col_tension, desc_tension in zip(slit_tension_cathode.keys(), slit_tension_cathode.values()) :
                data_transformed[f"{prefix}_Cathode_{desc_speed}x{desc_tension}"] = (
                    data_transformed[col_speed] * data_transformed[col_tension]
                )
        
        
        return data_transformed


    def _dv_17(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV17_'
        data_transformed = data

        anode_thk = data["X_PV_Assemble_Winding_WINDING_fdcwnd5506"]
        cathode_thk = data["X_PV_Assemble_Winding_WINDING_fdcwnd5497"]

        data_transformed[f"{prefix}__양극 - 음극 두께"] = (anode_thk - cathode_thk)

        return data_transformed

    def _dv_20(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV20_'
        data_transformed = data

        for col_speed, desc_speed in zip(COLS_WINDING_SPEED.keys(), COLS_WINDING_SPEED.values()):
            for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                 data_transformed[f"{prefix}_{desc_speed}x{desc_tension}"] = (
                    data_transformed[col_speed] * data_transformed[col_tension]
                )

        return data_transformed

    def _dv_27(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV27_'
        data_transformed = data

        data_transformed[f"{prefix}_square"] = data_transformed['X_PV_Assemble_Assembly_CBD_END_fdccbd0002']**2
        data_transformed[f"{prefix}_log"] = np.log(data_transformed['X_PV_Assemble_Assembly_CBD_END_fdccbd0002'])

        return data_transformed

    def _dv_29(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV29_'
        data_transformed = data

        def tmp(x):
            try : 
                x = x[0]
            except : 
                pass
            return x
        data_transformed['06_Assembly_Lot ID'] = data_transformed['06_Assembly_Lot ID'].apply(lambda x : tmp(x))

        

        avg_weight_by_lot = (
            data_transformed
            .groupby('06_Assembly_Lot ID')
            ['X_PV_Assemble_Assembly_ELF_END_fdcelf0008'].mean()
            .to_frame()
            .rename(columns={'X_PV_Assemble_Assembly_ELF_END_fdcelf0008':'avg_weight_by_lot'})
        )

        data_transformed = (
            data_transformed
            .merge(
                avg_weight_by_lot,
                how='left',
                left_on='06_Assembly_Lot ID',
                right_on='06_Assembly_Lot ID'
            )
        )
        data_transformed[f'{prefix}_Weight Deviation'] = data_transformed['X_PV_Assemble_Assembly_ELF_END_fdcelf0008'] - data_transformed['avg_weight_by_lot']

        return data_transformed

    def _dv_30(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV30_'
        data_transformed = data

        data_transformed[f"{prefix}_Charge_03_SEI"] = (
            data_transformed['X_NFF_Charge_03_Charge_Timemins']
            * data_transformed['X_NFF_Charge_03_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_02 SEI"] = (
            data_transformed['X_NFF_Charge_02_Charge_Timemins']
            * data_transformed['X_NFF_Charge_02_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_03 SEI"] = (
            data_transformed['X_NFF_Charge_03_Charge_Timemins']
            * data_transformed['X_NFF_Charge_03_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_04 SEI"] = (
            data_transformed['X_NFF_Charge_04_Charge_Timemins']
            * data_transformed['X_NFF_Charge_04_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_05 SEI"] = (
            data_transformed['X_NFF_Charge_05_Charge_Timemins']
            * data_transformed['X_NFF_Charge_05_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_06 SEI"] = (
            data_transformed['X_NFF_Charge_06_Charge_Timemins']
            * data_transformed['X_NFF_Charge_06_Average_Temperature']
        )
        data_transformed[f"{prefix}_Charge_07 SEI"] = (
            data_transformed['X_NFF_Charge_07_Charge_Timemins']
            * data_transformed['X_NFF_Charge_07_Average_Temperature']
        )

        # data_transformed[f"{prefix}_High Temp Aging_03 SEI"] = (
        #     data_transformed['X_NFF_High Temp Aging_03_Charge_Timemins']
        #     * data_transformed['X_NFF_High Temp Aging_03_Average_Temperature']
        # )
        # data_transformed[f"{prefix}_High Temp Aging_02 SEI"] = (
        #     data_transformed['X_NFF_High Temp Aging_02_Charge_Timemins']
        #     * data_transformed['X_NFF_High Temp Aging_02_Average_Temperature']
        # )

        return data_transformed

    def _dv_35(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV35_'
        data_transformed = data

        for col_line_speed in COLS_LIINE_SPEED : 
            for col_temp, desc_temp in zip(COLS_COATING_DRY_TEMPERATURE.keys(), COLS_COATING_DRY_TEMPERATURE.values()):
                for col_fan, desc_fan in zip(COLS_COATING_FAN.keys(), COLS_COATING_FAN.values()) :
                    data_transformed[f"{prefix}_{desc_temp}x{desc_fan}/{col_line_speed}"] = (
                        data_transformed[col_temp] * data_transformed[col_fan] / data_transformed[col_line_speed]
                    )

        return data_transformed

    def _dv_36(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV36_'
        data_transformed = data

        for col_rp_thickness, desc_rp_thickness in zip(COLS_ROLL_PRESSING_THICKNESS.keys(), COLS_ROLL_PRESSING_THICKNESS.values()):
            for col_slt_thickness, desc_slt_thickness in zip(COLS_SLITTING_THICKNESS.keys(), COLS_SLITTING_THICKNESS.values()) :
                data_transformed[f"{prefix}_{desc_slt_thickness}-{desc_rp_thickness}"] = (
                    data_transformed[col_slt_thickness] - data_transformed[col_rp_thickness]
                )

        return data_transformed

    def _dv_40(self, data:pd.DataFrmae) -> pd.DataFrame:
        prefix = 'DV40_'
        data_transformed = data

        col_start = '01_Mixing_Finished Date'
        col_end = '07_Before Degas_Finished Date'

        def tmp(x):
            try : 
                x = x[0]
            except : 
                pass
            return x
        try : 

            display(data_transformed[col_start])
            display(data_transformed[col_end])

            data_transformed[col_start] = pd.to_datetime(data_transformed[col_start].apply(lambda x : tmp(x)), errors='coerce')
            data_transformed[col_end] = pd.to_datetime(data_transformed[col_end].apply(lambda x : tmp(x)), errors='coerce')

            data_transformed[f"{prefix}_{col_end}-{col_start}"] = ((
                data_transformed[col_end] - data_transformed[col_start]
            ).dt.total_seconds()//3600)
        except : 
            pass
        return data_transformed

    def _dv_42(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV40_'
        data_transformed = data

        # Anode

        col_vis = 'y_GJJJ_Electrode_Mixing_Anode_점도_Inspection Measure Value'
        col_solid = 'y_GJJJ_Electrode_Mixing_Anode_고형분_Inspection Measure Value'
        #col_gap = 'Y_Anode_SPC_y_Coating_절연 갭'

        for col_temp, desc_temp in zip(COLS_COATING_DRY_TEMPERATURE.keys(), COLS_COATING_DRY_TEMPERATURE.values()) :
            data_transformed[f"{prefix}_Anode_점도*고형분 + 코팅갭*{col_temp}"] = (
                (
                    data_transformed[col_vis]
                    * data_transformed[col_solid]
                ) +
                (
                    #data_transformed[col_gap]
                    #* 
                    data_transformed[col_temp]
                )
            )

        # Cathode

        col_vis = 'y_GJJJ_Electrode_Mixing_Cathode_점도_Inspection Measure Value'
        col_solid = 'y_GJJJ_Electrode_Mixing_Cathode_고형분_Inspection Measure Value'
        #col_gap = 'Y_Cathode_SPC_y_Coating_절연 갭'

        for col_temp, desc_temp in zip(COLS_COATING_DRY_TEMPERATURE.keys(), COLS_COATING_DRY_TEMPERATURE.values()) :
            data_transformed[f"{prefix}_Cathode_점도*고형분 + 코팅갭*{col_temp}"] = (
                (
                    data_transformed[col_vis]
                    * data_transformed[col_solid]
                ) +
                (
                    #data_transformed[col_gap]
                    #* 
                    data_transformed[col_temp]
                )
            )

        return data_transformed


    def _dv_47(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV47_'
        data_transformed = data

        list_candidates = [
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5001": "분리막(상) 수동 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5029": "양극 수동 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5042": "음극 수동 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5015": "분리막(하) 수동 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5029": "양극 수동 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5042": "음극 수동 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5002": "분리막(상) 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5030": "양극 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5043": "음극 투입 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5016": "분리막(하) 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5030": "양극 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5043": "음극 투입 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5004": "분리막(상) 1단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5032": "양극 1단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5045": "음극 1단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5018": "분리막(하) 1단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5032": "양극 1단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5045": "음극 1단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5005": "분리막(상) 1단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5033": "양극 1단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5046": "음극 1단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5019": "분리막(하) 1단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5033": "양극 1단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5046": "음극 1단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5007": "분리막(상) 2단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5035": "양극 2단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5048": "음극 2단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5021": "분리막(하) 2단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5035": "양극 2단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5048": "음극 2단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5008": "분리막(상) 2단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5036": "양극 2단 완료 장력",   
                "X_PV_Assemble_Winding_WINDING_fdcwnd5049": "음극 2단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5022": "분리막(하) 2단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5036": "양극 2단 완료 장력",   
                "X_PV_Assemble_Winding_WINDING_fdcwnd5049": "음극 2단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5009": "분리막(상) 3단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5037": "양극 3단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5050": "음극 3단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5023": "분리막(하) 3단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5037": "양극 3단 시작 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5050": "음극 3단 시작 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5010": "분리막(상) 3단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5038": "양극 3단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5051": "음극 3단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5024": "분리막(하) 3단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5038": "양극 3단 완료 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5051": "음극 3단 완료 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5013": "분리막(상) 커팅 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5040": "양극 커팅 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5053": "음극 커팅 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5027": "분리막(하) 커팅 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5040": "양극 커팅 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5053": "음극 커팅 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5014": "분리막(상) 커팅 후 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5041": "양극 커팅 후 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5054": "음극 커팅 후 투입 장력",
            },
            {
                "X_PV_Assemble_Winding_WINDING_fdcwnd5028": "분리막(하) 커팅 후 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5041": "양극 커팅 후 투입 장력",
                "X_PV_Assemble_Winding_WINDING_fdcwnd5054": "음극 커팅 후 투입 장력",
            }
        ]

        for col_dict in list_candidates :
            col_1, desc_1 = list(col_dict.keys())[0], list(col_dict.values())[0]
            col_2, desc_2 = list(col_dict.keys())[1], list(col_dict.values())[1]
            col_3, desc_3 = list(col_dict.keys())[2], list(col_dict.values())[2]
            data_transformed[f"{prefix}_max({desc_1}, {desc_2}, {desc_3})-min"] = (
                pd.concat([data_transformed[col_1], data_transformed[col_2], data_transformed[col_3]], axis=1).max(axis=1)
                - pd.concat([data_transformed[col_1], data_transformed[col_2], data_transformed[col_3]], axis=1).max(axis=1)
            )

        return data_transformed

    def _dv_52(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV52_'
        data_transformed = data

        anode_thk = data["X_PV_Assemble_Winding_WINDING_fdcwnd5506"]
        cathode_thk = data["X_PV_Assemble_Winding_WINDING_fdcwnd5497"]
        sep_thk = data['X_PV_Assemble_Winding_WINDING_fdcwnd5515']
        turn_count = data['X_PV_Assemble_Winding_WINDING_fdcwnd5467']

        data_transformed[f"{prefix}__두께 x 회전수"] = (anode_thk + cathode_thk + sep_thk) * turn_count

        return data_transformed

    def _dv_56(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV52_'
        data_transformed = data

        col_1 = data["X_IQC_Electrode_Mixing_Cathode_금속이물_Cu_Zn"]
        col_2 = data["X_IQC_Electrode_Mixing_Cathode_금속이물_Fe_Cr"]
        col_3= data['X_IQC_Electrode_Mixing_Cathode_자성이물']

        data_transformed[f"{prefix}_Cathode_이물 총합"] = col_1 + col_2 + col_3

        
        col_1 = data["X_IQC_Electrode_Mixing_Anode_금속이물_Cu_Zn"]
        col_2 = data["X_IQC_Electrode_Mixing_Anode_금속이물_Fe_Cr"]
        col_3= data['X_IQC_Electrode_Mixing_Anode_자성이물']

        data_transformed[f"{prefix}_Anode_이물 총합"] = col_1 + col_2 + col_3

        return data_transformed


    
    #hs 추가   # 실제 주액량/설정 주액량
    def _dv_200(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV200_'
        data_transformed = data
        results = []
        
        for col_elf, desc_elf in zip(COLS_ASSEMBLY_ELF_AMOUNT.keys(), COLS_ASSEMBLY_ELF_AMOUNT.values()) :
            for col_elf_sv in COLS_ASSEMBLY_ELF_AMOUNT_SV :
                
                print(col_loading, desc_loading, col_elf_sv)
                try : 

                    # )
                    ratio = (
                        data[col_elf] / data[col_elf_sv]
                    )
                    results.append((f"{prefix}_{desc_elf}/{col_elf_sv}", ratio))
                except : 
                    pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed
    
    #hs 추가   # 음극+양극
    def _dv_210(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV210_'
        data_transformed = data
        results = []
        
        for col_cathod_thickness in COLS_ELECTRODE_CATHOD_THICKNESS :
            for col_anode_thickness in COLS_ELECTRODE_ANODE_THICKNESS :
                
                print(col_cathod_thickness, col_anode_thickness)
                try : 

                    # )
                    total = (
                        data[col_cathod_thickness] + data[col_anode_thickness]
                    )
                    results.append((f"{prefix}_{col_cathod_thickness}/{col_anode_thickness}", total))
                except : 
                    pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed
    
    #hs 추가   # 최내곽 곡률응력 - 전극두께/권심사이즈
    def _dv_212(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV212_'
        data_transformed = data
        results = []
        
        for col_cathod_width in COLS_ELECTRODE_CATHOD_WIDTH :
            for col_anode_width in COLS_ELECTRODE_ANODE_WIDTH :
                for col_winding_core, desc_winding_core in zip(COLS_WINDING_CORE.keys(), COLS_WINDING_CORE.values()) :
                
                    print(col_cathod_width, col_anode_width, col_winding_core, desc_winding_core)
                    try : 

                        # )
                        total = (
                            (data[col_cathod_width] + data[col_anode_width])/(data[col_widing_core]/2)
                        )
                        results.append((f"{prefix}_{col_cathod_width}/{col_anode_width}", total))
                    except : 
                        pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_83(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix ='DV83_'
        data_transformed = data
        # TODO : Coating -> Roll Pressing으로 명칭 변경
        data_transformed[f"{prefix}__음극 x 양극 Coating 로딩"] = (
            data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩']
            * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩']
        )
        # data_transformed[f"{prefix}__음극 x 양극 Coating 총로딩량1"] = (
        #     data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩_총 로딩량1']
        #     * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩_총 로딩량1']   
        # )
        # data_transformed[f"{prefix}__음극 x 양극 Coating 총로딩량2"] = (
        #     data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩_총 로딩량2']
        #     * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩_총 로딩량2']   
        # )
        data_transformed[f"{prefix}__음극 x 양극 Roll Pressing 로딩"] = (
            data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩']
            * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩']
        )
        # data_transformed[f"{prefix}__음극 x 양극 Roll Pressing 총로딩량1"] = (
        #     data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩_총 로딩량1']
        #     * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩_총 로딩량1']   
        # )
        # data_transformed[f"{prefix}__음극 x 양극 Roll Pressing 총로딩량2"] = (
        #     data_transformed['y_LQC_Electrode_Roll Pressing_Anode_로딩_총 로딩량2']
        #     * data_transformed['y_LQC_Electrode_Roll Pressing_Cathode_로딩_총 로딩량2']   
        # )

        return data_transformed

    def _dv_84(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix ='DV84_'
        data_transformed = data

        data_transformed[f"{prefix}__Anode 점도 x 라인 속도"] = (
            data_transformed['y_GJJJ_Electrode_Mixing_Anode_점도_Inspection Measure Value']
            * data_transformed['X_PV_Electrode_RollPress_Anode_D0006004_F_DN_Mean']
        )
        data_transformed[f"{prefix}__Cathode 점도 x 라인 속도"] = (
            data_transformed['y_GJJJ_Electrode_Mixing_Cathode_점도_Inspection Measure Value']
            * data_transformed['X_PV_Electrode_RollPress_Cathode_D0006004_F_DN_Mean']
        )

        return data_transformed

    def _dv_85(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV85_'
        data_transformed = data
        '''
        "FDCWND5012": "분리막(상) 권취 감속시점 장력",
        "FDCWND5026": "분리막(하) 권취 감속시점 장력",
        "FDCWND5039": "양극 권취 감속위치 장력",
        "FDCWND5052": "음극 권취 감속위치 장력",
        "FDCWND5466": "P1 최종 권취 감속도",
        '''

        cols_tension = {
            'X_PV_Assemble_Winding_WINDING_fdcwnd5012' : "WINDING_1 분리막(상) 권취 감속시점 장력",
            'X_PV_Assemble_Winding_WINDING_02_fdcwnd5012' : "WINDING_2 분리막(상) 권취 감속시점 장력",
            'X_PV_Assemble_Winding_WINDING_fdcwnd5026' : "WINDING_1 분리막(하) 권취 감속시점 장력",
            'X_PV_Assemble_Winding_WINDING_02_fdcwnd5026' : "WINDING_2 분리막(하) 권취 감속시점 장력",
            'X_PV_Assemble_Winding_WINDING_fdcwnd5039' : "WINDING_1 양극 권취 감속위치 장력",
            'X_PV_Assemble_Winding_WINDING_02_fdcwnd5039' : "WINDING_1 양극 권취 감속위치 장력",
            'X_PV_Assemble_Winding_WINDING_fdcwnd5052' : "WINDING_2 음극 권취 감속위치 장력",
            'X_PV_Assemble_Winding_WINDING_02_fdcwnd5052' : "WINDING_2 음극 권취 감속위치 장력"

        }

        cols_speed = {
            'X_PV_Assemble_Winding_WINDING_fdcwnd5466' : "WINDING_1 P1 최종 권취 감속도",
            'X_PV_Assemble_Winding_WINDING_02_fdcwnd5466' : "WINDING_2 P1 최종 권취 감속도", 
        }

        for col_tension, desc_tension in zip(cols_tension.keys(), cols_tension.values()):
            for col_speed, desc_speed in zip(cols_speed.keys(), cols_speed.values()) :
                data_transformed[f"{prefix}_{desc_tension} x {desc_speed}"] = (
                    data_transformed[col_tension] 
                    * data_transformed[col_speed]
                )

        return data_transformed

    def _dv_86(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV86_'
        data_transformed = data

        for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()):
            data_transformed[f"{prefix}_{desc_tension} * 가속도 WINDING 1"] = (
                data_transformed[col_tension] *
                data_transformed['X_PV_Assemble_Winding_WINDING_fdcwnd5461']
            )
            data_transformed[f"{prefix}_{desc_tension} * 가속도 WINDING 2"] = (
                data_transformed[col_tension] *
                data_transformed['X_PV_Assemble_Winding_WINDING_02_fdcwnd5461']
            )

        return data_transformed

    def _dv_88(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV88'
        data_transformed = data

        cols_1 = [
            'X_PV_Assemble_Assembly_ABW_END_fdcacw0005',
            'X_PV_Assemble_Assembly_CSZ_END_fdcacw0005',
            'X_PV_Assemble_Assembly_CRW_END_fdcacw0005',
            'X_PV_Assemble_Assembly_CBD_END_fdcacw0005',
            'X_PV_Assemble_Assembly_CCR_END_fdcacw0005',
            'X_PV_Assemble_Assembly_ELF_END_fdcacw0005'
        ]
        cols_2 = [
        'X_PV_Assemble_Assembly_ABW_END_fdcacw0006',
        'X_PV_Assemble_Assembly_CSZ_END_fdcacw0006',
        'X_PV_Assemble_Assembly_CRW_END_fdcacw0006',
        'X_PV_Assemble_Assembly_CBD_END_fdcacw0006',
        'X_PV_Assemble_Assembly_CCR_END_fdcacw0006',
        'X_PV_Assemble_Assembly_ELF_END_fdcacw0006',
        ]
        descs_process =[
            'ABW',
            'CSZ',
            'CRW',
            'CBD',
            'CCR',
            'ELF'
        ]
        for col_time, col_voltage, desc_process in zip(
            cols_1,
            cols_2,
            descs_process
        ) :
            data_transformed[f"{prefix}_{desc_process}_용접 시간xVoltage"] = (
                data_transformed[col_time]
                * data_transformed[col_voltage]
            )
        return data_transformed

