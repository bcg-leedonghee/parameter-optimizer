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
    "X_PV_Assembly_ELF_END_fdcelf0117": "함침 챔버 STEP#1 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0118": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0119": "함침 챔버 STEP#1 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0130": "함침 챔버 STEP#2 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0131": "함침 챔버 STEP#2 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0143": "함침 챔버 STEP#3 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0144": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0145": "함침 챔버 STEP#3 공정 서보밸브 압력(Kpa&Mpa) End",
    "X_PV_Assembly_ELF_END_fdcelf0156": "함침 챔버 STEP#4 공정 압력(Kpa&Mpa",
    "X_PV_Assembly_ELF_END_fdcelf0157": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) Start",
    "X_PV_Assembly_ELF_END_fdcelf0158": "함침 챔버 STEP#4 공정 서보밸브 압력(Kpa&Mpa) End",
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

COLS_WINDING_SPEED = {
    'X_PV_Winding_fdcwnd5460': "P1 1단 권취 속도",
    'X_PV_Winding_fdcwnd5463': "P1 2단 권취 속도",
    'X_PV_Winding_fdcwnd5465': "P1 3단 권취 속도",
    'X_PV_Winding_fdcwnd5466': "P1 분리막 잔량 권취 속도"

}

COLS_WASHING_TEMPERATURE = [
  'X_SV_Washing_Parameter Spec LSL_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Parameter Spec LSL_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Washing_Parameter Spec LSL_2차 저장탱크 히터 SV값',
  'X_SV_Washing_Parameter Spec LSL_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Parameter Spec LSL_3차 건조 핫부스터#4 온도 SV값',
  'X_SV_Washing_Parameter Spec LSL_순간온수기 온도 설정값',
  'X_SV_Washing_Parameter Spec USL_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_3차 건조 핫부스터#4 온도 SV값',
 'X_SV_Washing_Parameter Spec USL_순간온수기 온도 설정값',
 'X_SV_Washing_Parameter Value_2차 저장탱크 히터 SV값',
 'X_SV_Washing_Paramter Target Value_1차 저장탱크 히터 SV값',
  'X_SV_Washing_Paramter Target Value_2차 건조 핫부스터#1 온도 SV값',
 'X_SV_Washing_Paramter Target Value_2차 건조 핫부스터#2 온도 SV값',
 'X_SV_Washing_Paramter Target Value_2차 저장탱크 히터 SV값',
  'X_SV_Washing_Paramter Target Value_3차 건조 핫부스터#3 온도 SV값',
 'X_SV_Washing_Paramter Target Value_3차 건조 핫부스터#4 온도 SV값',
  'X_SV_Washing_Paramter Target Value_순간온수기 온도 설정값',
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

class FeatureGenerator:
    def __init__(self):
        pass

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:

        # 원본 데이터 복사
        data_transformed = data.copy()

        # DV02 : 주액 압력 / 와인더 텐션
        data_transformed = (
            self._dv_02(data_transformed)
        )

        # DV03 : 주앱 압력 * 주액 시간 / 와인더 텐션
        data_transformed = (
            self._dv_03(data_transformed)
        )

        # DV06 : 주앱 압력 *  워싱 온도 / 와인더 텐션
        data_transformed = (
            self._dv_06(data_transformed)
        )

        #DV07 : 주앱 압력 * 주액 시간 x 워싱 온도 / 와인더 텐션
        data_transformed = (
            self._dv_07(data_transformed)
        )

        # DV15 : 활성화 공정 종료 시간 - 조립 공정 종료 시간
        data_transformed = (
            self._dv_15(data_transformed)
        )

        # DV11 : 워싱 온도 / 와인더 텐션
        data_transformed = (
            self._dv_11(data_transformed)
        )

        # DV13 : 전극 Lot 전환 플래그
        data_transformed = (
            self._dv_13(data_transformed)
        )

        DV16 : 슬리팅 이물 플래그
        data_transformed = (
            self._dv_16(data_transformed)
        )

        # DV17 : 양극 음극 두께 잔차
        data_transformed = (
            self._dv_17(data_transformed)
        )

        # DV20 : 와인딩 속도 x 텐션 tradeoff
        data_transformed = (
            self._dv_20(data_transformed)
        )

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
                           print(e)
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
                    print(e)
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
            "X_Anode_PV_Slitting_D0006204_N" : "Line Speed SV"
        }
        slit_speed_cathode = {
            "X_Cathode_PV_Slitting_D0006204_N" : "Line Speed SV"
        }
        slit_tension_anode = {
            'X_Anode_PV_Slitting_R00155_N' : "Unwinder Tension SV",
            "X_Anode_PV_Slitting_R00158_N": "TM Outfeed Tension SV",
            "X_Anode_PV_Slitting_R00159_N": "Rewinder Upper Tension Taper",
            "X_Anode_PV_Slitting_R00160_N": "Rewinder Lower Tension Taper",
            "X_Anode_PV_Slitting_R00161_N": "Rewinder Upper Tension SV",
            "X_Anode_PV_Slitting_R00162_N": "Rewinder Lower Tension SV",
        }
        slit_tension_cathode = {
            'X_Cathode_PV_Slitting_R00155_N' : "Unwinder Tension SV",
            "X_Cathode_PV_Slitting_R00158_N": "TM Outfeed Tension SV",
            "X_Cathode_PV_Slitting_R00159_N": "Rewinder Upper Tension Taper",
            "X_Cathode_PV_Slitting_R00160_N": "Rewinder Lower Tension Taper",
            "X_Cathode_PV_Slitting_R00161_N": "Rewinder Upper Tension SV",
            "X_Cathode_PV_Slitting_R00162_N": "Rewinder Lower Tension SV",
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

        anode_thk = data["X_PV_Winding_fdcwnd5506"]
        cathode_thk = data["X_PV_Winding_fdcwnd5497"]

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

