# isort:imports-stdlib
from __future__ import annotations
from typing import Dict, List, Optional

# isort:imports-thirdparty
import numpy as np
import pandas as pd

# isort:imports-firstparty

# isort:import-localfolder

# =============================================================================
# 1) 의미키 -> 컬럼명 매핑
# =============================================================================
FEATURE_MAP: Dict[str, str] = {
    "cell_id":  "07_Before Degas_Cell ID",
    "lot_id":  "06_Assembly_Lot ID",
    # 와인딩
    "winding_speed":"X_PV_Winding_FDCWND5460","anode_spool_tension":"X_PV_Winding_FDCWND5059",
    "anode_thickness":"X_PV_Winding_FDCWND5506","cathode_thickness":"X_PV_Winding_FDCWND5497",
    "anode_width":"X_PV_Winding_FDCWND5507","anode_preform_y":"X_PV_Winding_FDCWND5437",
    "airjet_form_end":"X_PV_Winding_FDCWND5470","winding_total_len":"X_PV_Winding_FDCWND5203",
    # 조립 - 용접/성형/주액
    "weld_peak_power_pv":"X_PV_Assembly_FDCCRWH101","ccw_weld_power":"X_PV_Assembly_FDCCCW0005",
    "acw_weld_power":"X_PV_Assembly_FDCACW0005","ccw_weld_time":"X_PV_Assembly_FDCCCW0006",
    "ccw_weld_count":"X_PV_Assembly_FDCCCW0003","cbd_head_total_cnt":"X_PV_Assembly_FDCCBD0002",
    "elf_proc_time_s":"X_PV_Assembly_FDCELF0003","elf_pump_base_g":"X_PV_Assembly_FDCELF0005",
    "elf_final_dose_g":"X_PV_Assembly_FDCELF0008",
    # 조립 - Can/J/R 치수 (placeholder - 실제 컬럼명 필요)
    "can_inner_diameter":"X_PV_Assembly_CAN_INNER_DIAMETER",
    "jr_outer_diameter":"X_PV_Winding_JR_OUTER_DIAMETER",
    "negative_line_number":"X_PV_Assembly_NEGATIVE_LINE_NUMBER",  # placeholder
    # 전극 - 두께/압연
    "press1_thickness":"X_PV_Electrode_W715F","press2_thickness":"X_PV_Electrode_W795F",
    "slitting_thickness":"X_PV_Electrode_W155F","thk_cs_ds_1":"X_PV_Electrode_D9501",
    "roll_gap_1_ds":"X_PV_Electrode_D0037110","rolling_force":"X_PV_Electrode_D0003590",
    "main_roll_usage":"X_PV_Electrode_R03680",
    # 전극 - 믹싱
    "solid_content":"X_PV_Electrode_D10964","solid_content_avg":"X_PV_Electrode_D10968",
    "mixer_motor_amp":"X_PV_Electrode_D12280","active_wt_sv":"X_PV_Electrode_R1000",
    "active_wt_pv":"X_PV_Electrode_R1006",
    # 전극 - 코팅
    "coat_line_speed":"X_PV_Electrode_D2500","dry_temp_mid_top":"X_PV_Electrode_W2204",
    "supply_fan_rpm":"X_PV_Electrode_D2720","coat_gap_sv":"X_PV_Electrode_R13026",
    "coat_pump_rpm_pv":"X_PV_Electrode_R13015","chamber_humidity":"X_PV_Electrode_W220B",
    "ng_unfilled_wrinkle_top":"X_PV_Electrode_R12304","ng_unfilled_wrinkle_back":"X_PV_Electrode_R12319",
    # 전극 - 로딩/폭/정렬
    "loading_total":"X_PV_Electrode_D2266","electrode_width":"X_PV_Electrode_D20800",
    "mismatch_top1":"X_PV_Electrode_W1390","mismatch_top2":"X_PV_Electrode_W1392",
    "anode_loading":"X_PV_Electrode_ANODE_LOADING",  # placeholder
    "cathode_loading":"X_PV_Electrode_CATHODE_LOADING",  # placeholder
    # 전극 - 슬리팅/재권취
    "slit_line_speed":"X_PV_Electrode_R332","slit_stretch_tension":"X_PV_Electrode_D6012",
    "slit_knife_usage_1":"X_PV_Electrode_R24016","rewinder_length":"X_PV_Electrode_D6132",
    # 활성화 - 함침/주액/워싱 (1~12 번용)
    "inject_pressure":"X_PV_Activation_INJ_PRESS","inject_time":"X_PV_Activation_INJ_TIME",
    "inject_amount":"X_PV_Activation_INJ_AMOUNT","washing_temp":"X_PV_Activation_WASH_TEMP",
    "washing_water_temp":"X_PV_Activation_WASH_WATER_TEMP","activation_wait_time":"X_PV_Activation_WAIT_TIME",
    # 활성화 - 시간/대기
    "assembly_end_time":"X_PV_Timestamp_ASSEMBLY_END","activation_start_time":"X_PV_Timestamp_ACTIVATION_START",
    "aging_time":"X_PV_Activation_AGE_TIME","aging_avg_temp":"X_PV_Activation_AGE_TEMP",
    "activation_end_time":"X_PV_Timestamp_ACTIVATION_END",  # placeholder
    # 저항/IR
    "activation_acir":"X_PV_Activation_ACIR","washing_ir":"X_PV_Washing_IR",
    # 전극 LOT
    "electrode_lot":"X_PV_Electrode_LOT_ID","anode_lot":"X_PV_Electrode_ANODE_LOT",
    "cathode_lot":"X_PV_Electrode_CATHODE_LOT",
    # 원자재/이물
    "anode_material":"X_PV_Material_ANODE","cathode_material":"X_PV_Material_CATHODE",
    "conductor_material":"X_PV_Material_CONDUCTOR","electrolyte_material":"X_PV_Material_ELECTROLYTE",
    "can_material":"X_PV_Material_CAN",
    # Hi-pot/이물 지표
    "foreign_anode":"X_PV_Inspection_FOREIGN_ANODE","foreign_cathode":"X_PV_Inspection_FOREIGN_CATHODE",
    "foreign_conductor":"X_PV_Inspection_FOREIGN_CONDUCTOR","foreign_electrolyte":"X_PV_Inspection_FOREIGN_ELECTROLYTE",
    "foreign_can":"X_PV_Inspection_FOREIGN_CAN",
    "foreign_hopper":"X_PV_Inspection_FOREIGN_HOPPER","foreign_winding":"X_PV_Inspection_FOREIGN_WINDING",
    "foreign_welding":"X_PV_Inspection_FOREIGN_WELDING",
    # 전해액 첨가제
    "electrolyte_additive_low_voltage":"X_PV_Material_ADDITIVE_LV",
    # 웹 보정 (placeholder)
    "web_correction_rate":"X_PV_Winding_WEB_CORRECTION_RATE",
    # 슬리팅 시간 (placeholder)
    "slitting_completion_time":"X_PV_Timestamp_SLITTING_COMPLETE",
    "winding_start_time":"X_PV_Timestamp_WINDING_START",
    # 결함 플래그 (placeholder)
    "defect_flag":"Y_Defect_Flag",
    # 믹싱 시간 (placeholder)
    "mixing_start_time":"X_PV_Timestamp_MIXING_START",
}

def _lower_code(v):
    head, _, tail = v.rpartition("_")
    return f"{head}_{tail.lower()}" if head else v.lower()

FEATURE_MAP = {k: _lower_code(v) for k, v in FEATURE_MAP.items()}

DEFAULT_CONSTS = {
    "thickness_match_target_gap": 0.0, 
    "specific_gravity_target": 1.0, 
    "design_width": 0.0,
    # 활성화 관련 상수
    "target_impregnation_time": 60.0,  # 기준 함침 소요시간 [hr]
    "np_ratio_target": 1.05,  # N/P 비율 목표
}


class FeatureGenerator:
    """Defect feature generation with pandas DataFrame operations."""
    
    def __init__(self):
        self.consts = DEFAULT_CONSTS.copy()
        self.lot_keys = {}
    
    def fit(self, data: pd.DataFrame) -> None:
        """No fitting required for rule-based features."""
        pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from data using pandas operations."""
        ctx = {**DEFAULT_CONSTS, **self.consts}
        ctx["_lot_electrode"] = self.lot_keys.get("electrode")
        ctx["_lot_assembly"] = self.lot_keys.get("assembly")
        
        out = pd.DataFrame(index=data.index)
        
        # Cell ID
        out = self._add_cell_id(data, out)
        
        # 1~12. 활성화 공정 features (DV61~DV72)
        out = self._add_activation_impregnation_features(data, out, ctx)
        
        # 1. 와인딩 - 사선주름/사행불량
        out = self._add_winding_sagun_features(data, out, ctx)
        
        # 2. 와인딩 - FDF 불량
        out = self._add_winding_fdf_features(data, out, ctx)
        
        # 3. 와인딩 - 외경/Can 삽입불량
        out = self._add_winding_outer_diameter_features(data, out, ctx)
        
        # 4. 조립 - 약용접
        out = self._add_assembly_welding_features(data, out, ctx)
        
        # 5. 조립 - 성형불량
        out = self._add_assembly_forming_features(data, out, ctx)
        
        # 6. 조립 - 주액불량
        out = self._add_assembly_electrolyte_features(data, out, ctx)
        
        # 8. 횡단 - Cross-process Cascade
        out = self._add_crossprocess_cascade_features(data, out, ctx)
        
        # 9. 전극 - 로딩/용량
        out = self._add_electrode_loading_features(data, out, ctx)
        
        # 10. 전극 - R/P 두께
        out = self._add_electrode_thickness_features(data, out, ctx)
        
        # 11. 전극 - 미스매치/Camber
        out = self._add_electrode_mismatch_features(data, out, ctx)
        
        # 12. 전극 - Foil Burr
        out = self._add_electrode_burr_features(data, out, ctx)
        
        # 13. 전극 - 접착력
        out = self._add_electrode_adhesion_features(data, out, ctx)
        
        # 14. 전극 - 수분
        out = self._add_electrode_moisture_features(data, out, ctx)
        
        # 16. 전극 - 믹싱·코팅·압연 조합
        out = self._add_electrode_mixing_coating_features(data, out, ctx)
        
        # 17. 전극 - 재권취→형교환
        out = self._add_electrode_rewinding_features(data, out, ctx)
        
        # 18. 횡단 - 추가
        out = self._add_crossprocess_extra_features(data, out, ctx)

        out = pd.concat([data, out], axis=1)
        
        return out
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def set_constants(self, consts: Dict[str, float]) -> None:
        """Set constant values for feature calculation."""
        self.consts.update(consts)
    
    def set_lot_keys(self, lot_keys: Dict[str, str]) -> None:
        """Set LOT key column names for grouping."""
        self.lot_keys.update(lot_keys)
    
    # =============================================================================
    # 헬퍼 함수들
    # =============================================================================
    
    def _get_col(self, df: pd.DataFrame, key: str) -> pd.Series:
        """Get column by feature key, return numeric series."""
        name = FEATURE_MAP.get(key, key)
        if name in df.columns:
            s = df[name]
            return pd.to_numeric(s, errors="coerce") if s.dtype == object else s
        return pd.Series(np.nan, index=df.index)
    
    def _get_auto_key(self, df: pd.DataFrame) -> Optional[str]:
        """Get automatic LOT key column name."""
        for k in ("coating_lot", "lot_id"):
            if k in df.columns:
                return k
        return None
    
    def _lot_std(self, df: pd.DataFrame, s: pd.Series, key: Optional[str] = None) -> pd.Series:
        """Calculate group-wise standard deviation within LOT."""
        key = key or self._get_auto_key(df)
        if not key or key not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return s.groupby(df[key]).transform("std")
    
    def _lot_mean(self, df: pd.DataFrame, s: pd.Series, key: Optional[str] = None) -> pd.Series:
        """Calculate group-wise mean within LOT."""
        key = key or self._get_auto_key(df)
        if not key or key not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return s.groupby(df[key]).transform("mean")
    
    def _lot_z(self, df: pd.DataFrame, s: pd.Series, key: Optional[str] = None) -> pd.Series:
        """Calculate z-score within LOT, fallback to global z-score if no LOT."""
        key = key or self._get_auto_key(df)
        if not key or key not in df.columns:
            mu, sd = s.mean(), s.std()
            return (s - mu) / (sd if sd else np.nan)
        g = s.groupby(df[key])
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        return (s - mu) / sd
    
    def _zmean(self, df: pd.DataFrame, cols: List[pd.Series], key: Optional[str] = None) -> pd.Series:
        """Calculate mean of z-scores across multiple columns."""
        z_scores = pd.concat([self._lot_z(df, c, key) for c in cols], axis=1)
        return z_scores.mean(axis=1)
    
    def _u_flag(self, s: pd.Series, lo: float = 0.1, hi: float = 0.9) -> pd.Series:
        """Flag values in the lower or upper quantile range (U-shaped)."""
        if s.notna().sum() == 0:
            return pd.Series(pd.NA, index=s.index, dtype="Int8")
        a, b = s.quantile(lo), s.quantile(hi)
        return ((s <= a) | (s >= b)).astype("Int8")
    
    def _lin_resid(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Calculate linear regression residual: y - (a*x + b)."""
        m = y.notna() & x.notna()
        if m.sum() < 2:
            return pd.Series(np.nan, index=y.index)
        a, b = np.polyfit(x[m].astype(float), y[m].astype(float), 1)
        return y - (a * x + b)
    
    def _calc_time_diff(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
        """Calculate time difference between two timestamp columns in hours."""
        start = self._get_col(df, start_col)
        end = self._get_col(df, end_col)
        try:
            start_dt = pd.to_datetime(start, errors='coerce')
            end_dt = pd.to_datetime(end, errors='coerce')
            return (end_dt - start_dt).dt.total_seconds() / 3600
        except:
            return pd.Series(np.nan, index=df.index)
    
    def _calc_line_position_encoding(self, df: pd.DataFrame, line_col: str) -> pd.Series:
        """Encode production line position (even/odd parity)."""
        line_num = self._get_col(df, line_col)
        if line_num.notna().sum() == 0:
            return pd.Series(np.nan, index=df.index)
        return (line_num.astype(int) % 2).astype("Int8")
    
    # =============================================================================
    # Feature addition methods
    # =============================================================================
    
    def _add_activation_impregnation_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """1~12. 활성화 공정 - 함침/저전압/용량 features (DV61~DV72).
        
        새로 추가된 feature 번호:
        - DV61: 와인더 텐션 기반 함침 저해
        - DV62: 주액 추진력/텐션 비율
        - DV63: 주액 개별 인자 (3 개: 압력, 시간, 양)
        - DV64: 주액×워싱 온도/텐션 복합
        - DV65: 활성화 공정 대기시간
        - DV66: 주액×워싱 온도 (텐션 제외)
        - DV67: 워싱액 온도/텐션 비율
        - DV68: 워싱수 온도
        - DV69: 전극 LOT 전환 플래그
        - DV70: 전극 원자재 특성 편차 (2 개: 양극/음극 LOT 변동성)
        - DV71: 조립→활성화 대기시간
        - DV72: Hi-pot 누적 위험 스코어
        - DV73: SEI 형성 누적 노출량
        - DV74: ACIR-IR 갭
        - DV75: 트레이 위치 인코딩
        """
        
        # DV61__와인더 텐션 기반 함침 저해
        tension = self._get_col(data, "anode_spool_tension")
        out["DV61__와인더 텐션 기반 함침 저해"] = tension
        
        # DV62__주액 추진력/텐션 비율
        pressure = self._get_col(data, "inject_pressure")
        inj_time = self._get_col(data, "inject_time")
        amount = self._get_col(data, "inject_amount")
        propulsion = pressure * inj_time * amount
        out["DV62__주액 추진력/텐션 비율"] = propulsion / tension.replace(0, np.nan)
        
        # DV63__주액 개별 인자 (3 개)
        out["DV63__주액 압력"] = pressure
        out["DV63__주액 시간"] = inj_time
        out["DV63__주액 양"] = amount
        
        # DV64__주액×워싱 온도/텐션 복합
        washing_temp = self._get_col(data, "washing_temp")
        out["DV64__주액×워싱 온도/텐션 복합"] = (propulsion * washing_temp) / tension.replace(0, np.nan)
        
        # DV65__활성화 공정 대기시간
        wait_time = self._get_col(data, "activation_wait_time")
        out["DV65__활성화 공정 대기시간"] = wait_time
        
        # DV66__주액×워싱 온도 (텐션 제외)
        out["DV66__주액×워싱 온도"] = propulsion * washing_temp
        
        # DV67__워싱액 온도/텐션 비율
        out["DV67__워싱액 온도/텐션 비율"] = washing_temp / tension.replace(0, np.nan)
        
        # DV68__워싱수 온도
        washing_water_temp = self._get_col(data, "washing_water_temp")
        out["DV68__워싱수 온도"] = washing_water_temp
        
        # DV69__전극 LOT 전환 플래그
        electrode_lot = self._get_col(data, "electrode_lot")
        lot_change_flag = (electrode_lot != electrode_lot.shift(1)).astype("Int8")
        n_cells = 50
        out["DV69__전극 LOT 전환 플래그"] = lot_change_flag.rolling(window=n_cells, min_periods=1).max()
        
        # DV70__전극 원자재 특성 편차 (2 개)
        anode_lot = self._get_col(data, "anode_lot")
        cathode_lot = self._get_col(data, "cathode_lot")
        out["DV70__양극 LOT 변동성"] = anode_lot.astype(str).groupby((anode_lot != anode_lot.shift()).cumsum()).transform('count')
        out["DV70__음극 LOT 변동성"] = cathode_lot.astype(str).groupby((cathode_lot != cathode_lot.shift()).cumsum()).transform('count')
        
        # DV71__조립→활성화 대기시간
        assembly_wait = self._calc_time_diff(data, "assembly_end_time", "activation_start_time")
        out["DV71__조립→활성화 대기시간"] = assembly_wait
        
        # DV72__Hi-pot 누적 위험 스코어
        foreign_anode = self._get_col(data, "foreign_anode")
        foreign_cathode = self._get_col(data, "foreign_cathode")
        foreign_conductor = self._get_col(data, "foreign_conductor")
        foreign_electrolyte = self._get_col(data, "foreign_electrolyte")
        foreign_can = self._get_col(data, "foreign_can")
        foreign_hopper = self._get_col(data, "foreign_hopper")
        foreign_winding = self._get_col(data, "foreign_winding")
        foreign_welding = self._get_col(data, "foreign_welding")
        
        material_foreign = (foreign_anode + foreign_cathode + foreign_conductor + 
                           foreign_electrolyte + foreign_can) / 5
        process_foreign = (foreign_hopper + foreign_winding + foreign_welding) / 3
        
        out["DV72__Hi-pot 위험 스코어"] = material_foreign + process_foreign
        
        # DV73__SEI 형성 누적 노출량 (Row 30 - 추가)
        aging_time = self._get_col(data, "aging_time")
        aging_temp = self._get_col(data, "aging_avg_temp")
        out["DV73__SEI 형성 누적 노출량"] = aging_time * aging_temp
        
        # DV74__ACIR-IR 갭 (Row 31 - 추가)
        acir = self._get_col(data, "activation_acir")
        washing_ir = self._get_col(data, "washing_ir")
        out["DV74__ACIR-IR 갭"] = (acir - washing_ir).abs()
        
        # DV75__트레이 위치 인코딩 (Row 32 - 추가)
        # TODO: 실제 트레이 위치 컬럼명이 확인되면 FEATURE_MAP 에 추가 필요
        out["DV75__트레이 위치 인코딩"] = pd.Series(np.nan, index=data.index)
        
        return out
    
    def _add_cell_id(self, data: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """Add Cell ID feature."""
        cid = FEATURE_MAP["cell_id"]
        if cid in data.columns:
            out["cell_id"] = data[cid]
        elif "cell_id" in data.columns:
            out["cell_id"] = data["cell_id"]
        return out
    
    def _add_winding_sagun_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """1. 와인딩 - 사선주름/사행불량 features (5 개 + 추가)."""
        anode_thk = self._get_col(data, "anode_thickness")
        cathode_thk = self._get_col(data, "cathode_thickness")
        out["DV01__양극 - 음극 두께 매칭 잔차"] = (anode_thk - cathode_thk) - ctx["thickness_match_target_gap"]
        
        loading = self._get_col(data, "loading_total")
        out["DV01__너울 프록시 (로딩편차)"] = self._lot_std(data, loading, ctx.get("_lot_electrode"))
        
        winding_speed = self._get_col(data, "winding_speed")
        tension = self._get_col(data, "anode_spool_tension")
        out["DV01__텐션 - 권취속도 trade-off 지수"] = winding_speed / tension.replace(0, np.nan)
        
        anode_width = self._get_col(data, "anode_width")
        out["DV01__단위면적당 권취 힘"] = tension / anode_width.replace(0, np.nan)
        
        mismatch = self._get_col(data, "mismatch_top1")
        out["DV01__캠버×권취속도 교호항"] = mismatch * winding_speed
        
        # DV01__캠버×대기시간 교호항 (Row 14 - 추가)
        camber_wait_time = self._calc_time_diff(data, "slitting_completion_time", "winding_start_time")
        camber_value = self._lot_mean(data, mismatch, ctx.get("_lot_electrode"))
        out["DV01__캠버×대기시간 교호항"] = camber_value * camber_wait_time
        
        # DV01__사행보정률 변동성 (Row 17 - 추가)
        web_correction_rate = self._get_col(data, "web_correction_rate")
        out["DV01__사행보정률 변동성"] = self._lot_std(data, web_correction_rate, ctx.get("_lot_electrode"))
        
        return out
    
    def _add_winding_fdf_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """2. 와인딩 - FDF 불량 features (3 개)."""
        airjet = self._get_col(data, "airjet_form_end")
        out["DV02__프리포밍 각도 편차"] = airjet - 60.0
        
        wrinkle_top = self._get_col(data, "ng_unfilled_wrinkle_top")
        wrinkle_back = self._get_col(data, "ng_unfilled_wrinkle_back")
        out["DV02__무지부 주름 cascade flag"] = wrinkle_top + wrinkle_back
        
        preform_y = self._get_col(data, "anode_preform_y")
        out["DV02__주행롤러 - 잎롤러 거리×각도"] = preform_y * airjet
        
        return out
    
    def _add_winding_outer_diameter_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """3. 와인딩 - 외경/Can 삽입불량 features (1 개)."""
        anode_thk = self._get_col(data, "anode_thickness")
        cathode_thk = self._get_col(data, "cathode_thickness")
        out["DV03__누적 두께 합 (4 겹)"] = 2 * anode_thk + 2 * cathode_thk
        
        return out
    
    def _add_assembly_welding_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """4. 조립 - 약용접 features (6 개)."""
        ccw_power = self._get_col(data, "ccw_weld_power")
        ccw_time = self._get_col(data, "ccw_weld_time")
        out["DV04__인가에너지 효율"] = ccw_power * ccw_time
        
        ccw_count = self._get_col(data, "ccw_weld_count")
        out["DV04__혼 수명 누적 타수"] = ccw_count
        
        out["DV04__강도 - 손상 trade-off 위치"] = self._lot_z(data, ccw_power, ctx.get("_lot_assembly"))
        
        acw_power = self._get_col(data, "acw_weld_power")
        out["DV04__양극 - 음극 용접 비대칭"] = ccw_power - acw_power
        
        # DV04__용접 SV-PV 갭 (Row 23 - 추가)
        weld_peak_power_pv = self._get_col(data, "weld_peak_power_pv")
        ccw_weld_power_sv = self._get_col(data, "ccw_weld_power")  # SV 로 가정
        out["DV04__용접 SV-PV 갭"] = (ccw_weld_power_sv - weld_peak_power_pv).abs()
        
        # DV04__원부자재 LOT 전환 플래그 (Row 25 - 추가)
        conductor_lot = self._get_col(data, "conductor_material")
        lot_change_flag = (conductor_lot != conductor_lot.shift(1)).astype("Int8")
        n_cells = 50
        out["DV04__원부자재 LOT 전환 플래그"] = lot_change_flag.rolling(window=n_cells, min_periods=1).max()
        
        return out
    
    def _add_assembly_forming_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """5. 조립 - 성형불량 features (3 개)."""
        cbd_count = self._get_col(data, "cbd_head_total_cnt")
        out["DV05__금형 누적 성형 카운트"] = cbd_count
        
        out["DV05__카운트 비선형 구간 플래그"] = self._u_flag(cbd_count)
        
        # DV05__레인 패리티 인코딩 (Row 20 - 추가)
        # 레인 번호가 있는 경우 홀짝 인코딩, 없으면 NaN
        # TODO: 실제 레인 번호 컬럼명이 확인되면 FEATURE_MAP 에 추가 필요
        line_num = self._get_col(data, "negative_line_number")  # placeholder
        if not line_num.isna().all():
            out["DV05__레인 패리티 인코딩"] = self._calc_line_position_encoding(data, "negative_line_number")
        else:
            out["DV05__레인 패리티 인코딩"] = pd.Series(np.nan, index=data.index)
        
        return out
    
    def _add_assembly_electrolyte_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """6. 조립 - 주액불량 features (4 개)."""
        elf_time = self._get_col(data, "elf_proc_time_s")
        out["DV06__J/R 조립공정 체류시간"] = elf_time
        
        elf_dose = self._get_col(data, "elf_final_dose_g")
        out["DV06__주액 비중 환산 편차"] = elf_dose / (ctx["specific_gravity_target"] or np.nan)
        
        out["DV06__주액량 outlier score"] = self._lot_z(data, elf_dose, ctx.get("_lot_assembly"))
        
        # DV06__Gap 밴드 이탈 플래그 (Row 22 - 추가)
        # J/R 외경 - Can 내경 갭이 0.2~0.5mm 밴드를 벗어나는지 확인
        # TODO: 실제 Can 내경 컬럼명이 확인되면 FEATURE_MAP 에 추가 필요
        can_inner_diameter = self._get_col(data, "can_inner_diameter")  # placeholder
        jr_outer_diameter = self._get_col(data, "jr_outer_diameter")  # placeholder
        gap = jr_outer_diameter - can_inner_diameter
        gap_band_violation = ((gap < 0.2) | (gap > 0.5)).astype("Int8")
        out["DV06__Gap 밴드 이탈 플래그"] = gap_band_violation
        
        return out
    
    def _add_crossprocess_cascade_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """8. 횡단 - Cross-process Cascade features (1 개)."""
        wrinkle_top = self._get_col(data, "ng_unfilled_wrinkle_top")
        wrinkle_back = self._get_col(data, "ng_unfilled_wrinkle_back")
        airjet = self._get_col(data, "airjet_form_end")
        out["DV08__무지부주름→FDF→사행 전파 체인"] = (wrinkle_top + wrinkle_back) * (airjet - 60.0).abs()
        
        return out
    
    def _add_electrode_loading_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """9. 전극 - 로딩/용량 features (6 개)."""
        loading = self._get_col(data, "loading_total")
        out["DV09__로딩 LOT 내 변동성"] = self._lot_std(data, loading, ctx.get("_lot_electrode"))
        
        active_sv = self._get_col(data, "active_wt_sv")
        active_pv = self._get_col(data, "active_wt_pv")
        out["DV09__활물질 중량 SV-PV 갭"] = active_sv - active_pv
        
        pump_rpm = self._get_col(data, "coat_pump_rpm_pv")
        out["DV09__Pump RPM 비선형 보정항"] = pump_rpm ** 2
        
        gap_sv = self._get_col(data, "coat_gap_sv")
        line_speed = self._get_col(data, "coat_line_speed")
        out["DV09__Coating Gap×Speed 교호"] = gap_sv * line_speed
        
        # DV09__N/P 비율 (Row 33 - 추가)
        # 음극 로딩 / 양극 로딩 (목표 >= 1.05)
        # TODO: 양극/음극 로딩 별도 컬럼이 확인되면 FEATURE_MAP 에 추가 필요
        anode_loading = self._get_col(data, "anode_loading")  # placeholder
        cathode_loading = self._get_col(data, "cathode_loading")  # placeholder
        np_ratio = anode_loading / cathode_loading.replace(0, np.nan)
        out["DV09__N/P 비율"] = np_ratio
        out["DV09__N/P 부족 플래그"] = (np_ratio < ctx.get("np_ratio_target", 1.05)).astype("Int8")
        
        return out
    
    def _add_electrode_thickness_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """10. 전극 - R/P 두께 features (3 개)."""
        press2_thk = self._get_col(data, "press2_thickness")
        slitting_thk = self._get_col(data, "slitting_thickness")
        out["DV10__압연후 - 슬리팅후 두께변화"] = press2_thk - slitting_thk
        
        thk_cs_ds = self._get_col(data, "thk_cs_ds_1")
        out["DV10__두께 폭방향 불균일"] = thk_cs_ds
        
        press1_thk = self._get_col(data, "press1_thickness")
        roll_gap = self._get_col(data, "roll_gap_1_ds")
        out["DV10__Roll Gap 선형 잔차"] = self._lin_resid(press1_thk, roll_gap)
        
        return out
    
    def _add_electrode_mismatch_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """11. 전극 - 미스매치/Camber features (3 개)."""
        mismatch1 = self._get_col(data, "mismatch_top1")
        out["DV11__CPC 중심정렬 잔차"] = mismatch1
        
        mismatch2 = self._get_col(data, "mismatch_top2")
        out["DV11__양극 - 음극 camber 합성"] = np.sqrt(mismatch1 ** 2 + mismatch2 ** 2)
        
        electrode_width = self._get_col(data, "electrode_width")
        winding_len = self._get_col(data, "winding_total_len")
        out["DV11__폭 편차 누적 (권취길이 가중)"] = (electrode_width - ctx["design_width"]) * winding_len
        
        return out
    
    def _add_electrode_burr_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """12. 전극 - Foil Burr features (3 개)."""
        slit_speed = self._get_col(data, "slit_line_speed")
        slit_tension = self._get_col(data, "slit_stretch_tension")
        out["DV12__Burr 위험지수"] = slit_speed * slit_tension
        
        knife_usage = self._get_col(data, "slit_knife_usage_1")
        out["DV12__칼날 누적 슬리팅 길이"] = knife_usage
        
        out["DV12__양극 burr 가중 플래그"] = knife_usage * 1.5
        
        return out
    
    def _add_electrode_adhesion_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """13. 전극 - 접착력 features (2 개)."""
        dry_temp = self._get_col(data, "dry_temp_mid_top")
        fan_rpm = self._get_col(data, "supply_fan_rpm")
        line_speed = self._get_col(data, "coat_line_speed")
        migration = dry_temp * fan_rpm / line_speed.replace(0, np.nan)
        out["DV13__Binder migration 지수"] = migration
        
        loading = self._get_col(data, "loading_total")
        out["DV13__로딩×건조강도 교호"] = loading * migration
        
        return out
    
    def _add_electrode_moisture_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """14. 전극 - 수분 features (2 개)."""
        dry_temp = self._get_col(data, "dry_temp_mid_top")
        slit_speed = self._get_col(data, "slit_line_speed")
        out["DV14__R2R 건조 충분도"] = dry_temp / slit_speed.replace(0, np.nan)
        
        humidity = self._get_col(data, "chamber_humidity")
        out["DV14__라인 dew point(습도) 프록시"] = humidity
        
        return out
    
    def _add_electrode_mixing_coating_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """16. 전극 - 믹싱·코팅·압연 조합 features (4 개)."""
        solid_content = self._get_col(data, "solid_content")
        rolling_force = self._get_col(data, "rolling_force")
        out["DV16__고형분 × 압연력 교호"] = solid_content * rolling_force
        
        solid_avg = self._get_col(data, "solid_content_avg")
        out["DV16__점도 in-spec 거리"] = (solid_content - solid_avg).abs()
        
        gap_sv = self._get_col(data, "coat_gap_sv")
        out["DV16__3 공정 조건 클러스터 ID"] = self._zmean(data, [solid_content, gap_sv, rolling_force], ctx.get("_lot_electrode"))
        
        motor_amp = self._get_col(data, "mixer_motor_amp")
        out["DV16__모터 부하 사후지표"] = motor_amp
        
        return out
    
    def _add_electrode_rewinding_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """17. 전극 - 재권취→형교환 features (1 개)."""
        rewinder_len = self._get_col(data, "rewinder_length")
        out["DV17__형교환 빈도 추정"] = 1.0 / rewinder_len.replace(0, np.nan)
        
        return out
    
    def _add_crossprocess_extra_features(self, data: pd.DataFrame, out: pd.DataFrame, ctx: Dict) -> pd.DataFrame:
        """18. 횡단 - 추가 features (7 개)."""
        loading = self._get_col(data, "loading_total")
        out["DV18__LOT 통계 fallback(로딩 std)"] = self._lot_std(data, loading, ctx.get("_lot_electrode"))
        
        roll_usage = self._get_col(data, "main_roll_usage")
        out["DV18__설비 누적 사용량"] = roll_usage
        
        pump_base = self._get_col(data, "elf_pump_base_g")
        elf_dose = self._get_col(data, "elf_final_dose_g")
        out["DV18__SV-PV 갭 패밀리 (주액)"] = pump_base - elf_dose
        
        # DV18__SV-PV 갭 패밀리 (전공정) (Row 39 - 추가)
        # 주요 CTP 파라미터들의 SV-PV 갭 합산
        sv_pv_gaps = []
        
        # 용접 SV-PV 갭
        weld_peak_power_pv = self._get_col(data, "weld_peak_power_pv")
        ccw_weld_power_sv = self._get_col(data, "ccw_weld_power")
        sv_pv_gaps.append((ccw_weld_power_sv - weld_peak_power_pv).abs())
        
        # 주액 SV-PV 갭
        sv_pv_gaps.append((pump_base - elf_dose).abs())
        
        # 코팅 갭 SV-PV (있다면)
        coat_gap_sv = self._get_col(data, "coat_gap_sv")
        # coat_gap_pv = self._get_col(data, "coat_gap_pv")  # TODO: 실제 컬럼명 추가
        # sv_pv_gaps.append((coat_gap_sv - coat_gap_pv).abs())
        
        if sv_pv_gaps:
            out["DV18__SV-PV 갭 패밀리 (전공정)"] = pd.concat(sv_pv_gaps, axis=1).mean(axis=1)
        else:
            out["DV18__SV-PV 갭 패밀리 (전공정)"] = pd.Series(np.nan, index=data.index)
        
        # DV18__총 리드타임 (Row 40 - 추가)
        # 믹싱 시작 → 활성화 종료까지 총 시간
        # TODO: 실제 타임스탬프 컬럼명이 확인되면 FEATURE_MAP 에 추가 필요
        mixing_start_time = self._get_col(data, "mixing_start_time")  # placeholder
        activation_end_time = self._get_col(data, "activation_end_time")  # placeholder
        total_lead_time = self._calc_time_diff(data, "mixing_start_time", "activation_end_time")
        out["DV18__총 리드타임"] = total_lead_time
        
        # DV18__자기상관 (직전 셀) (Row 41 - 추가)
        # 동일 설비에서 직전 셀의 결과 (lag feature)
        # TODO: 설비 ID 와 판정 컬럼이 확인되면 FEATURE_MAP 에 추가 필요
        defect_flag = self._get_col(data, "defect_flag")  # placeholder
        if not defect_flag.isna().all():
            out["DV18__자기상관 (직전 셀)"] = defect_flag.shift(1)
        else:
            out["DV18__자기상관 (직전 셀)"] = pd.Series(np.nan, index=data.index)
        
        return out