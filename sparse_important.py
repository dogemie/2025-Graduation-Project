import os
import shutil
import pathlib
import itertools
import pandas as pd
import pycolmap

# 📌 COLMAP 작업 경로 설정
output_path = pathlib.Path("output")  # COLMAP 결과 저장 폴더
match_db_path = output_path / "match_db"  # Feature Matching 결과 폴더
sparse_output_path = output_path / "sparse"  # Sparse Reconstruction 결과 저장 폴더
if sparse_output_path.exists():
    shutil.rmtree(sparse_output_path)
sparse_output_path.mkdir(exist_ok=True)

# ✅ 사용된 데이터베이스 (Feature Matching 후 결과)
database_path = match_db_path / "matched_database_0.db"
image_dir = pathlib.Path("images")  # COLMAP에서 사용한 이미지 폴더

# ✅ 1️⃣ **튜닝할 변수 리스트 생성**
min_num_matches_list = [13, 15, 17]  # ✅ 최소 매칭 개수
min_model_size_list = [11, 13, 15, 19]  # ✅ 최소 등록 이미지 개수
init_num_trials_list = [500, 1000, 1500]  # ✅ Reconstruction 초기화 시도 횟수

param_list = list(itertools.product(min_num_matches_list, min_model_size_list, init_num_trials_list))

# ✅ 2️⃣ **Sparse Reconstruction 실행 함수**
def run_sparse_reconstruction(i, min_num_matches, min_model_size, init_num_trials):
    exp_sparse_output_path = sparse_output_path / f"sparse_{i}"  # 개별 실험 폴더 생성
    exp_sparse_output_path.mkdir(exist_ok=True)

    print(f"🔍 [{i+1}] Sparse Reconstruction: min_matches={min_num_matches}, min_model_size={min_model_size}, init_trials={init_num_trials}")

    # ✅ Reconstruction 옵션 설정
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 8  # 다중 스레드 설정
    options.ba_local_max_num_iterations = 50  # Local Bundle Adjustment 최대 반복 횟수
    options.ba_global_max_num_iterations = 100  # Global Bundle Adjustment 최대 반복 횟수
    options.min_num_matches = min_num_matches  # ✅ 최소 매칭 개수
    options.min_model_size = min_model_size  # ✅ 최소 등록 이미지 개수
    options.init_num_trials = init_num_trials  # ✅ 초기 이미지 페어 선택 시도 횟수
    options.multiple_models = False  # ✅ 단일 모델 Reconstruction 수행

    try:
        # ✅ Sparse Reconstruction 실행
        reconstruction = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(exp_sparse_output_path),
            options=options
        )

        num_images_registered = max([reconstruction[i].num_reg_images() for i in reconstruction])

        # ✅ `cameras.bin` 파일 크기 확인
        camera_bin_path = exp_sparse_output_path / "sparse/0/cameras.bin"
        camera_bin_size = os.path.getsize(camera_bin_path) if camera_bin_path.exists() else 0

        for model_id, model in reconstruction.items():
            print(f"Model {model_id}: 등록된 이미지 개수 = {model.num_reg_images()}")

        print(f"✅ Reconstruction 완료! {num_images_registered}개 이미지 등록됨. 저장 경로: {exp_sparse_output_path}")

    except Exception as e:
        print(f"❌ Reconstruction 실패: {e}")
        num_images_registered = 0
        camera_bin_size = 0

    return [exp_sparse_output_path, min_num_matches, min_model_size, init_num_trials, num_images_registered, camera_bin_size]

# ✅ 3️⃣ **모든 조합 실행**
results = []
for i, params in enumerate(param_list):
    results.append(run_sparse_reconstruction(i, *params))

# ✅ 4️⃣ **결과를 CSV 파일로 저장**
results_df = pd.DataFrame(results, columns=["output_path", "min_num_matches", "min_model_size", "init_num_trials", "num_images_registered", "camera_bin_size"])
results_df.to_csv(sparse_output_path / "sparse_results.csv", index=False)

print("✅ 모든 Sparse Reconstruction 실험 완료! 결과 CSV 저장됨.")
