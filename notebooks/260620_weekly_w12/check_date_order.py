"""
parquet 파일의 날짜 순서 보장 여부 확인 스크립트
"""
import pandas as pd

# 일부만 로드 (nrows 대신 파라미터 없음 -> 전체. 샘플만 보려면 아래처럼)
data_path_candidates = [
    '/home/admin/Documents/98_model/notebooks/260714_feature_engineering_qa/feature_store_v10_n32s.parquet',
    '/data01/Documents/98_model/notebooks/260714_feature_engineering_qa/feature_store_v10_n32s.parquet',
    '/data/98_model/notebooks/260714_feature_engineering_qa/feature_store_v10_n32s.parquet',
]

data = None
for path in data_path_candidates:
    try:
        # 날짜 컬럼만 로드해서 빠르게 확인
        data = pd.read_parquet(path, columns=['06_Assembly_Finished Date'])
        print(f"로드 성공: {path}")
        print(f"총 행 수: {len(data)}")
        break
    except Exception as e:
        print(f"로드 실패: {path} -> {e}")

if data is None:
    raise RuntimeError("parquet 파일을 불러올 수 없습니다.")

col_date = '06_Assembly_Finished Date'
data[col_date] = pd.to_datetime(data[col_date])

# 앞뒤 날짜 샘플 출력
print("\n=== 앞 10행 날짜 ===")
print(data[col_date].head(10).to_string())

print("\n=== 뒤 10행 날짜 ===")
print(data[col_date].tail(10).to_string())

# 날짜 오름차순 여부 확인
is_sorted_asc = data[col_date].is_monotonic_increasing
is_sorted_desc = data[col_date].is_monotonic_decreasing

print(f"\n날짜 오름차순 정렬 여부: {is_sorted_asc}")
print(f"날짜 내림차순 정렬 여부: {is_sorted_desc}")

# 역전된 구간 개수 확인 (오름차순 기준)
n_inversions = (data[col_date].diff() < pd.Timedelta(0)).sum()
print(f"날짜가 역전된 행 수 (앞 행보다 이전 날짜인 경우): {n_inversions}")

if n_inversions > 0:
    print("\n[경고] 날짜 순서가 보장되지 않습니다. sort_values 필요!")
    # 역전 발생 위치 샘플 출력
    inversion_idx = data[data[col_date].diff() < pd.Timedelta(0)].head(5).index
    print("역전 발생 위치 샘플 (index):")
    print(data.loc[inversion_idx, col_date])
else:
    print("\n[OK] 날짜 순서가 오름차순으로 보장됩니다.")
