import os
import shutil
import pathlib
import itertools
import pandas as pd
import pycolmap
import sqlite3
import multiprocessing

# 📌 COLMAP 관련 경로 설정
output_path = pathlib.Path("output")  # COLMAP 결과 저장 폴더
match_db_path = output_path / "match_db"  # Feature Matching 결과 저장 폴더
match_db_path.mkdir(parents=True, exist_ok=True)  # 폴더 생성

feature_csv = output_path / "feature_analysis.csv"
matching_csv = match_db_path / "matching_analysis.csv"

# ✅ Feature Matching 변수 조합 (4×2×4 = 32개 실험)
max_features = 8192  # ✅ 고정
max_ratio_list = [0.6]  # ✅ 거리 비율 제한
guided_matching_list = [True]  # ✅ 추가 매칭 수행 여부
min_num_inliers_list = [15]  # ✅ 최소 inlier 개수

# ✅ 1️⃣ **최적의 DB 찾기**
df_features = pd.read_csv(feature_csv)
best_db_path = df_features.loc[df_features["keypoint_avg"].idxmax(), "db_path"]

if not os.path.exists(best_db_path):
    raise FileNotFoundError(f"❌ 최적 DB 파일이 존재하지 않습니다: {best_db_path}")

print(f"✅ 최적 DB 선택 완료: {best_db_path}")

# ✅ 2️⃣ **특이점 매칭 실행 함수**
def match_features(i, max_ratio, guided_matching, min_num_inliers):
    temp_db = match_db_path / f"matched_database_{i}.db"  # 📌 output/match_db/ 내부에 저장
    print(f"🔍 [{i+1}] Matching Features: max_features={max_features}, max_ratio={max_ratio}, guided={guided_matching}, min_inliers={min_num_inliers}")

    try:
        # ✅ 기존 DB 파일을 복사하여 매칭 작업 수행
        shutil.copy(best_db_path, temp_db)

        # ✅ Feature Matching 실행 (Exhaustive Matching 사용)
        pycolmap.match_exhaustive(
            database_path=str(temp_db),
            sift_options=pycolmap.SiftMatchingOptions(
                num_threads=8,
                max_ratio=max_ratio,  # ✅ 거리 비율 제한
                guided_matching=guided_matching,  # ✅ 추가 매칭 여부
            ),
            matching_options=pycolmap.ExhaustiveMatchingOptions(),  # ✅ 기본 매칭 옵션 사용
            verification_options=pycolmap.TwoViewGeometryOptions(
                min_num_inliers=min_num_inliers  # ✅ 최소 inlier 개수 설정
            ),
            device=pycolmap.Device("cpu")
        )

        # ✅ 매칭 결과 분석
        with sqlite3.connect(temp_db) as conn:
            df_matches = pd.read_sql_query("SELECT * FROM matches;", conn)
            match_avg = df_matches.shape[0] if not df_matches.empty else 0

    except Exception as e:
        print(f"❌ 매칭 실패: {e}")
        match_avg = 0

    return [temp_db, max_features, max_ratio, guided_matching, min_num_inliers, match_avg]

# ✅ 3️⃣ **병렬 처리 실행**
param_list = list(itertools.product(max_ratio_list, guided_matching_list, min_num_inliers_list))

if __name__ == "__main__":
    # ✅ 기존 output 폴더 삭제 후 생성
    if match_db_path.exists():
        shutil.rmtree(match_db_path)
    match_db_path.mkdir(exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(match_features, [(i, *params) for i, params in enumerate(param_list)])

    # ✅ 4️⃣ **결과 CSV 저장**
    matching_df = pd.DataFrame(results, columns=["db_path", "max_features", "max_ratio", "guided_matching", "min_num_inliers", "match_avg"])
    matching_df.to_csv(matching_csv, index=False)

    print("✅ Feature Matching 완료! 결과 CSV 저장됨.")
