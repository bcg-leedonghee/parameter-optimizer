# isort:imports-stdlib
from pathlib import Path

# isort:imports-thirdparty

# isort:imports-firstparty

# isort:import-localfolder

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
SRC_PATH = PROJECT_PATH / "src"

DATA_PATH_BY_VERSION = {
    "v1": "/home/admin/Documents/0_hw/notebooks/data_loader/MERGED_MIX_260603_FLAT_W_IS_TRAIN.parquet",
    "v2": "/home/admin/Documents/0_hw/notebooks/data_loader_cleaned/202508_202509/MERGED_ALL_FLAT_v1.parquet",
    "v3": "/home/admin/Documents/0_hw/notebooks/data_loader/MERGED_MIX_260616_FLAT_W_IS_TRAIN.parquet",
    "v4" : "/home/admin/Documents/0_hw/notebooks/data_loader_cleaned/202602_202603_v2/MERGED_MIX_260619_FLAT_W_IS_TRAIN.parquet",
    "v5" : "/home/admin/Documents/data_for_model/20260630_LongMart.parquet",
    "v6" : "/home/admin/Documents/data_for_model/20260701_LongMart.parquet",
    "v7" : "/home/admin/Documents/data_for_model/20260702_LongMart_1.parquet",
    "v8" : "/home/admin/Documents/98_model/data/LongMart_n32s_50000_final.parquet",
    "v9-vm2" : "/data01/Documents/98_model/data/260709_LongMart_n32s_50000.parquet",
    "v9" : "/home/admin/Documents/98_model/data/260709_LongMart_n32s_50000.parquet",
    "v10-n32s-vm2" : "/data01/Documents/98_model/data/260714_LongMart_n32s_50000.parquet",
    "v10-n33q-vm2" : "/data01/Documents/98_model/data/260714_LongMart_n33q.parquet"
} # 경로 최신버전 파일인지 확인

DATA_PATH_TRACE_BY_VERSION = {
    "2507_2512": "/home/admin/Documents/0_hw/notebooks/data/TRACEABILITY_250701_251231.parquet"
}

COLUMN_MAPPING_PATH_BY_TYPE_PROCESS = {
    "PV_Electrode": "/home/admin/Documents/0_hw/data/mapping/PV_Electrode_v3.json",
    "PV_Assembly": "/home/admin/Documents/0_hw/data/mapping/PV_Assembly.json",
    "PV_Winding": "/home/admin/Documents/0_hw/data/mapping/PV_Winding.json",
    "SV_Mixing": "/home/admin/Documents/0_hw/data/mapping/SV_Mixing.json",
    "SV_Coating": "/home/admin/Documents/0_hw/data/mapping/SV_Coating.json",
    "SV_Roll Pressing": "/home/admin/Documents/0_hw/data/mapping/SV_Roll Pressing.json",
    "SV_Slitting": "/home/admin/Documents/0_hw/data/mapping/SV_Slitting.json",
    "SV_Winding": "/home/admin/Documents/0_hw/data/mapping/SV_Winding.json",
    "SV_Assembly": "/home/admin/Documents/0_hw/data/mapping/SV_Assembly.json",
    "SV_Washing": "/home/admin/Documents/0_hw/data/mapping/SV_Washing.json",
}
