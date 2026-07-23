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
    "X_PV_Assembly_ELF_END_fdcelf0106": "주액 챔버 진공펌프 압력(Kpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0107": "주액 챔버 진공펌프 압력(Kpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0108": "주액 챔버 서보밸브 압력(Kpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0109": "주액 챔버 서보밸브 압력(Kpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0111": "함침 챔버 O-RING 진공 압력(Kpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0112": "함침 챔버 O-RING 진공 압력(Kpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0113": "함침 챔버 STEP#1 (가압&진공&미사용)(mode",
    "X_PV_Assembly_ELF_END_fdcelf0114": "함침 챔버 STEP#1 가압&진공 상승 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0115": "함침 챔버 STEP#1 가압&진공 유지 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0116": "함침 챔버 STEP#1 가압&진공 하강 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0117": "함침 챔버 STEP#1 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0118": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0119": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0126": "함침 챔버 STEP#2 (가압&진공&미사용)(mode",
    "X_PV_Assembly_ELF_END_fdcelf0127": "함침 챔버 STEP#2 가압&진공 상승 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0128": "함침 챔버 STEP#2 가압&진공 유지 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0129": "함침 챔버 STEP#2 가압&진공 하강 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0130": "함침 챔버 STEP#2 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0131": "함침 챔버 STEP#2 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0132": "함침 챔버 STEP#2 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0139": "함침 챔버 STEP#3 (가압&진공&미사용)(mode",
    "X_PV_Assembly_ELF_END_fdcelf0140": "함침 챔버 STEP#3 가압&진공 상승 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0141": "함침 챔버 STEP#3 가압&진공 유지 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0142": "함침 챔버 STEP#3 가압&진공 하강 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0143": "함침 챔버 STEP#3 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0144": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0145": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0152": "함침 챔버 STEP#4 (가압&진공&미사용)(mode",
    "X_PV_Assembly_ELF_END_fdcelf0153": "함침 챔버 STEP#4 가압&진공 상승 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0154": "함침 챔버 STEP#4 가압&진공 유지 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0155": "함침 챔버 STEP#4 가압&진공 하강 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0156": "함침 챔버 STEP#4 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0157": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0158": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0165": "함침 챔버 STEP#5 (가압&진공&미사용)(mode",
    "X_PV_Assembly_ELF_END_fdcelf0166": "함침 챔버 STEP#5 가압&진공 상승 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0167": "함침 챔버 STEP#5 가압&진공 유지 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0168": "함침 챔버 STEP#5 가압&진공 하강 시간(s",
    "X_PV_Assembly_ELF_END_fdcelf0169": "함침 챔버 STEP#5 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0170": "함침 챔버 STEP#5 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0171": "함침 챔버 STEP#5 공정 서보밸브 압력(Kpa&Mpa) End",
}

COLS_ASSEMBLY_INJECTION_TIME = {
    "X_PV_Assembly_fdcelf0166": "함침 챔버 STEP#5 가압&진공 상승 시간(s",
    "X_PV_Assembly_fdcelf0167": "함침 챔버 STEP#5 가압&진공 유지 시간(s",
    "X_PV_Assembly_fdcelf0168": "함침 챔버 STEP#5 가압&진공 하강 시간(s",
}

COLS_WINDING_TENSION = {
    "X_PV_Winding_fdcwnd5001": "분리막(상) 수동 장력",
    "X_PV_Winding_fdcwnd5002": "분리막(상) 투입 장력",
    "X_PV_Winding_fdcwnd5004": "분리막(상) 1단 시작 장력",
    "X_PV_Winding_fdcwnd5005": "분리막(상) 1단 완료 장력",
    "X_PV_Winding_fdcwnd5007": "분리막(상) 2단 시작 장력",
    "X_PV_Winding_fdcwnd5008": "분리막(상) 2단 완료 장력",
    "X_PV_Winding_fdcwnd5009": "분리막(상) 3단 시작 장력",
    "X_PV_Winding_fdcwnd5010": "분리막(상) 3단 완료 장력",
    "X_PV_Winding_fdcwnd5011": "분리막(상) 터렛 회전 장력",
    "X_PV_Winding_fdcwnd5012": "분리막(상) 권취 감속시점 장력",
    "X_PV_Winding_fdcwnd5013": "분리막(상) 커팅 장력",
    "X_PV_Winding_fdcwnd5014": "분리막(상) 커팅 후 투입 장력",
    "X_PV_Winding_fdcwnd5015": "분리막(하) 수동 장력",
    "X_PV_Winding_fdcwnd5016": "분리막(하) 투입 장력",
    "X_PV_Winding_fdcwnd5018": "분리막(하) 1단 시작 장력",
    "X_PV_Winding_fdcwnd5019": "분리막(하) 1단 완료 장력",
    "X_PV_Winding_fdcwnd5021": "분리막(하) 2단 시작 장력",
    "X_PV_Winding_fdcwnd5022": "분리막(하) 2단 완료 장력",
    "X_PV_Winding_fdcwnd5023": "분리막(하) 3단 시작 장력",
    "X_PV_Winding_fdcwnd5024": "분리막(하) 3단 완료 장력",
    "X_PV_Winding_fdcwnd5025": "분리막(하) 터렛 회전 장력",
    "X_PV_Winding_fdcwnd5026": "분리막(하) 권취 감속시점 장력",
    "X_PV_Winding_fdcwnd5027": "분리막(하) 커팅 장력",
    "X_PV_Winding_fdcwnd5028": "분리막(하) 커팅 후 투입 장력",
    "X_PV_Winding_fdcwnd5029": "양극 수동 장력",
    "X_PV_Winding_fdcwnd5030": "양극 투입 장력",
    "X_PV_Winding_fdcwnd5032": "양극 1단 시작 장력",
    "X_PV_Winding_fdcwnd5033": "양극 1단 완료 장력",
    "X_PV_Winding_fdcwnd5035": "양극 2단 시작 장력",
    "X_PV_Winding_fdcwnd5036": "양극 2단 완료 장력",
    "X_PV_Winding_fdcwnd5037": "양극 3단 시작 장력",
    "X_PV_Winding_fdcwnd5038": "양극 3단 완료 장력",
    "X_PV_Winding_fdcwnd5039": "양극 권취 감속위치 장력",
    "X_PV_Winding_fdcwnd5040": "양극 커팅 장력",
    "X_PV_Winding_fdcwnd5041": "양극 커팅 후 투입 장력",
    "X_PV_Winding_fdcwnd5042": "음극 수동 장력",
    "X_PV_Winding_fdcwnd5043": "음극 투입 장력",
    "X_PV_Winding_fdcwnd5045": "음극 1단 시작 장력",
    "X_PV_Winding_fdcwnd5046": "음극 1단 완료 장력",
    "X_PV_Winding_fdcwnd5048": "음극 2단 시작 장력",
    "X_PV_Winding_fdcwnd5049": "음극 2단 완료 장력",
    "X_PV_Winding_fdcwnd5050": "음극 3단 시작 장력",
    "X_PV_Winding_fdcwnd5051": "음극 3단 완료 장력",
    "X_PV_Winding_fdcwnd5052": "음극 권취 감속위치 장력",
    "X_PV_Winding_fdcwnd5053": "음극 커팅 장력",
    "X_PV_Winding_fdcwnd5054": "음극 커팅 후 투입 장력",
    "X_PV_Winding_fdcwnd5055": "양극 Buffer 자동 장력",
    "X_PV_Winding_fdcwnd5056": "양극 Buffer 수동 장력",
    "X_PV_Winding_fdcwnd5057": "음극 Buffer 자동 장력",
    "X_PV_Winding_fdcwnd5058": "음극 Buffer 수동 장력",
    "X_PV_Winding_fdcwnd5059": "양극 스풀 장력",
    "X_PV_Winding_fdcwnd5060": "양극 스풀 장력"
}

COLS_WASHING_TEMPERATURE = [
     'X_SV_Washing_Parameter Spec LSL_1차 저장탱크 온도 상한값',
 'X_SV_Washing_Parameter Spec LSL_1차 저장탱크 온도 하한값',
  'X_SV_Washing_Parameter Spec LSL_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Parameter Spec LSL_2차 건조 핫부스터#2 온도 SV값',
  'X_SV_Washing_Parameter Spec LSL_2차 저장탱크 온도 상한값',
 'X_SV_Washing_Parameter Spec LSL_2차 저장탱크 온도 하한값',
 'X_SV_Washing_Parameter Spec LSL_2차 저장탱크 히터 SV값',
  'X_SV_Washing_Parameter Spec LSL_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Parameter Spec LSL_3차 건조 핫부스터#4 온도 SV값',
  'X_SV_Washing_Parameter Spec LSL_순간온수기 온도 설정값',
   'X_SV_Washing_Parameter Spec USL_1차 저장탱크 온도 상한값',
 'X_SV_Washing_Parameter Spec USL_1차 저장탱크 온도 하한값',
  'X_SV_Washing_Parameter Spec USL_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_2차 건조 핫부스터#2 온도 SV값',
  'X_SV_Washing_Parameter Spec USL_2차 저장탱크 온도 상한값',
 'X_SV_Washing_Parameter Spec USL_2차 저장탱크 온도 하한값',
 'X_SV_Washing_Parameter Spec USL_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_3차 건조 핫부스터#4 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_순간온수기 온도 설정값',
  'X_SV_Washing_Parameter Value_2차 저장탱크 온도 상한값',
 'X_SV_Washing_Parameter Value_2차 저장탱크 온도 하한값',
 'X_SV_Washing_Parameter Value_2차 저장탱크 히터 SV값',
  'X_SV_Washing_Paramter Target Value_1차 저장탱크 온도 상한값',
 'X_SV_Washing_Paramter Target Value_1차 저장탱크 온도 하한값',
 'X_SV_Washing_Paramter Target Value_1차 저장탱크 히터 SV값',
  'X_SV_Washing_Paramter Target Value_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Paramter Target Value_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Washing_Paramter Target Value_2차 저장탱크 온도 상한값',
 'X_SV_Washing_Paramter Target Value_2차 저장탱크 온도 하한값',
 'X_SV_Washing_Paramter Target Value_2차 저장탱크 히터 SV값',
  'X_SV_Washing_Paramter Target Value_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Paramter Target Value_3차 건조 핫부스터#4 온도 SV값',
  'X_SV_Washing_Paramter Target Value_순간온수기 온도 설정값',
]

#hs 추가
COLS_ASSEMBLY_ELF_AMOUNT = {
    'X_PV_Assembly_ELF_END_fdcelf0008' : '최종 토출량(g)'
    }
COLS_ASSEMBLY_ELF_AMOUNT_SV = [
    'X_SV_Assembly_Paramter Target Value_[EL Filling Chamber] PUMP 주액량 설정값'
                               ]
#0624 변경
#hs 추가 
COLS_ELECTRODE_ANODE_THICKNESS = [
    'X_RollMap_Anode_THICK_AVG_2', 
    'X_RollMap_Anode_THICK_AVG_3',
    'X_RollMap_Anode_THICK_AVG_4',
]

#0624 변경
#hs 추가
COLS_ELECTRODE_CATHOD_THICKNESS = [
    'X_RollMap_Cathode_THICK_AVG_2',
    'X_RollMap_Cathode_THICK_AVG_3',
]

#0624 추가
#hs 추가 
COLS_WINDING_CORE = {'X_PV_Winding_fdcwnd5496' : '권심 사이즈'}
COLS_WINDING_ACC = {'X_PV_Winding_fdcwnd5466' : 'P1 최종 권취 감속도',
                    'X_PV_Winding_fdcwnd5461' : 'P1 1단 권취 가속도',
                    }
#0624 추가
#hs 추가 
COLS_ELECTRODE_ANODE_WIDTH = [
    'X_RollMap_Anode_NFF_SLT_COT_WIDTH_LEN_BACK', 
    'X_RollMap_Anode_NFF_SLT_COT_WIDTH_LEN_TOP'
    ]
#0624 추가
COLS_ELECTRODE_CATHODE_WIDTH =[
    'X_RollMap_Cathode_NFF_SLT_COT_WIDTH_LEN_BACK', 
    'X_RollMap_Cathode_NFF_SLT_COT_WIDTH_LEN_TOP'
    ]
#0624 추가
COLS_IQC_ = [
    'X_IQC_입도_D10', 
    'X_IQC_입도_D50', 
    'X_IQC_입도_D90', 
    'X_IQC_입도_Dmax'
]

class FeatureGenerator:
    def __init__(self):
        pass

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:

        # 원본 데이터 복사
        data_transformed = data.copy()

        # DV02 : 주액 압력 / 와인더 텐션
        # data_transformed = (
        #     self._dv_02(data_transformed)
        # )

        # DV03 : 주앱 압력 * 주액 시간 / 와인더 텐션
        # data_transformed = (
        #     self._dv_03(data_transformed)
        # )

        # DV06 : 주앱 압력 *  워싱 온도 / 와인더 텐션
        data_transformed = (
            self._dv_06(data_transformed)
        )

        # DV07 : 주앱 압력 * 주액 시간 x 워싱 온도 / 와인더 텐션
        # data_transformed = (
        #     self._dv_07(data_transformed)
        # )

        return data_transformed

    def _dv_02(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV02_'
        data_transformed = data
        results = []

        for col_pressure, desc_pressure in zip(COLS_ASSEMBLY_PRESSURE.keys(), COLS_ASSEMBLY_PRESSURE.values()) :
            for col_tension, desc_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                print(col_pressure, desc_pressure, col_tension, desc_tension)
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
                    try : 
                        # data_transformed[f"{prefix}_{desc_pressure}x_{col_washing_temp}/{desc_tension}"] = (
                        #     data[col_pressure] * data[col_washing_temp] / data[col_tension]
                        # )
                        ratio = (
                            data[col_pressure] * data[col_washing_temp] / data[col_tension]
                        )
                        results.append((f"{prefix}_{desc_pressure}x_{col_washing_temp}/{desc_tension}", ratio))
                    except Exception as e: 
                        print(e)
        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
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
                            results.apend((f"{prefix}_{desc_pressure}x_{desc_time}x_{col_washing_temp}/{desc_tension}", ratio))
                        except Exception as e: 
                           print(e)
        # 한 번에 병합
        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed

    def _dv_09(self, data:pd.DataFrmae) -> pd.DataFrame:
        prefix = 'DV09_'
        data_transformed = data

        col_start = '06_Assembly_Finished Date'
        col_end = '07_Before Degas_Finished Date'

        data_transformed[col_start] = pd.to_datetime(data_transformed[col_start])
        data_transformed[col_end] = pd.to_datetime(data_transformed[col_end])

        data_transformed[f"{prefix}_{col_end}-{col_start}"] = (
            data_transformed[col_end] - data_transformed[col_start]
        )
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
    def _dv_213(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV213_'
        data_transformed = data
        results = []
        
        for col_cathod_thickness in COLS_ELECTRODE_CATHOD_THICKNESS :
            for col_anode_thickness in COLS_ELECTRODE_ANODE_THICHNESS :
                for col_winding_core, desc_winding_core in zip(COLS_WINDING_CORE.keys(), COLS_WINDING_CORE.values()) :
                
                    print(col_cathod_thickness, col_anode_thickness, col_winding_core, desc_winding_core)
                    try : 

                        # )
                        total = (
                            (data[col_cathod_thickness] + data[col_anode_thickness])/(data[col_widing_core]/2)
                        )
                        results.append((f"{prefix}_{col_cathod_thickness}+{col_anode_thickness}/{col_widing_core}/2", total))
                    except : 
                        pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed
    
            #0624 추가 #hs 추가  # 텐션 ÷ (전극 두께 ÷ 전극 폭)
    def _dv_214(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV214_'
        data_transformed = data
        results = []
        
        for col_cathod_width in COLS_ELECTRODE_CATHOD_WIDTH :
            for col_anode_width in COLS_ELECTRODE_ANODE_WIDTH :
                for col_winding_tension, desc_winding_tension in zip(COLS_WINDING_TENSION.keys(), COLS_WINDING_TENSION.values()) :
                
                    print(col_cathod_width, col_anode_width, col_winding_tension, desc_winding_tension)
                    try : 

                        
                        ratio = (
                             data[col_winding_tension]
                             /((data[col_cathod_thickness] + data[col_anode_thickness])
                               /(data[col_cathod_width] + data[col_anode_width]))
                        )
                        results.append((f"{prefix}_{col_winding_tension}/{col_cathod_thickness}+{col_anode_thickness}/{col_cathod_width}+{col_anode_width}", ratio))
                    except : 
                        pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed
    
    #0624 추가 #입도 D90 − D10, Dmax ÷ D50, (D90 − D50) ÷ (D50 − D10)
    def _dv_217(self, data:pd.DataFrame) -> pd.DataFrame:
        prefix = 'DV217_'
        data_transformed = data
        results = []
        try:
            
            d10, d50, d90, dmax = COLS_IQC_
            
            a = data[d90] - data[d10]
            b = (data[d90]-data[d50]) / (data[d50]-data[d10])
            c = data[dmax]/data[d50]
            
            results. append((f"{prefix}_{d90}-{d10}",a))
            results. append((f"{prefix}_({d90}-{d50})/({d50}-{d10})",b))
            results. append((f"{prefix}_{dmax}-{d50}",c))
            
        except:
            pass

        if results:
            result_df = pd.DataFrame({k: v for k, v in results})
            data_transformed = pd.concat([data_transformed, result_df], axis=1)
        return data_transformed
    
        