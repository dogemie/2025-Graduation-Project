import os
import shutil
import pathlib
import itertools
import pandas as pd
import pycolmap
import multiprocessing
import sqlite3

# 📌 COLMAP 관련 경로 설정
image_dir = pathlib.Path("train")  # 배경이 제거된 이미지 폴더
output_path = pathlib.Path("output")  # COLMAP 결과 저장 폴더

# ✅ 로그 폴더 설정
log_dir = pathlib.Path("log")
log_dir.mkdir(parents=True, exist_ok=True)

# ✅ CSV 파일 설정
feature_csv = output_path / "feature_analysis.csv"

# ✅ SIFT 변수 조합 (특이점 검출)
# num_octaves_list = [6 + 1 * x for x in range(4)]
# edge_threshold_list = [5 + 2 * x for x in range(6)]
# peak_threshold_list = [0.005 - 0.0009 * x for x in range(5)]

num_octaves_list = [6]
edge_threshold_list = [15]
peak_threshold_list = [0.0014]

# ✅ 특이점 검출 실행 함수 (병렬 처리)
def extract_features(i, num_octaves, edge_threshold, peak_threshold):
    temp_db = output_path / f"database_{i}.db"
    print(f"🔍 [{i+1}] Running SIFT extraction: num_octaves={num_octaves}, edge_threshold={edge_threshold}, peak_threshold={peak_threshold}")

    try:
        # ✅ Feature Extraction 실행
        pycolmap.extract_features(
            database_path=str(temp_db),
            image_path=str(image_dir),
            camera_model="SIMPLE_RADIAL",
            sift_options=pycolmap.SiftExtractionOptions(
                num_threads=8,
                max_num_features=8192,
                peak_threshold=peak_threshold,
                num_octaves=num_octaves,
                edge_threshold=edge_threshold,
            ),
            device=pycolmap.Device("cpu")
        )

    except Exception as e:
        print(f"❌ Error in feature extraction {i+1}: {e}")

    return temp_db

# ✅ 멀티프로세싱 실행
if __name__ == "__main__":
    
    # ✅ 기존 output 폴더 삭제 후 생성
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)
    
    param_list = list(itertools.product(num_octaves_list, edge_threshold_list, peak_threshold_list))

    # 병렬 처리 설정 (CPU 개수만큼 병렬 실행)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        db_paths = pool.starmap(extract_features, [(i, *params) for i, params in enumerate(param_list)])

    # ✅ DB 존재 여부 확인 및 특이점 개수 평균 계산
    itteration = 0
    feature_data = []
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"⚠️ Warning: Database {db_path} not found, skipping...")
            continue

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # ✅ WAL 모드 활성화 (동시 접근 문제 해결)
            cursor.execute("PRAGMA journal_mode=WAL;")

            # ✅ 모든 keypoints 데이터 로드
            df_keypoints = pd.read_sql_query("SELECT * FROM keypoints;", conn)

            # ✅ keypoints 테이블의 rows 값들의 평균 계산
            average_keypoints = df_keypoints["rows"].sum() / df_keypoints["image_id"].nunique()

            # ✅ matches 테이블의 전체 매칭 개수 계산
            df_matches = pd.read_sql_query("SELECT * FROM matches;", conn)
            match_avg = df_matches.shape[0] if not df_matches.empty else 0

        feature_data.append([db_path, param_list[itteration][0], param_list[itteration][1], param_list[itteration][2] , round(average_keypoints, 4)])
        itteration += 1

    # ✅ 특이점 & 매칭 개수 CSV 저장
    feature_df = pd.DataFrame(feature_data, columns=["db_path", "num_octaves", "edge_threshold", "peak_threshold", "keypoint_avg"])
    feature_df.to_csv(feature_csv, index=False)

    print("✅ Feature extraction completed successfully!")
