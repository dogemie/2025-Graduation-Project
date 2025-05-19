import os
import shutil
import pathlib
import itertools
import pandas as pd
import pycolmap

# 📌 COLMAP 작업 경로 설정
output_path = pathlib.Path("output")
match_db_path = output_path / "match_db"
sparse_output_path = output_path / "sparse"
backup_image_dir = pathlib.Path("backup_images")
image_dir = pathlib.Path("train")

# ✅ sparse 폴더 초기화
if sparse_output_path.exists():
    shutil.rmtree(sparse_output_path)
sparse_output_path.mkdir(exist_ok=True)

# ✅ 사용된 데이터베이스
database_path = match_db_path / "matched_database_0.db"

# ✅ 튜닝할 변수
min_num_matches_list = [15]
min_model_size_list = [11, 15, 19]
init_num_trials_list = [1000, 1500]

param_list = list(itertools.product(min_num_matches_list, min_model_size_list, init_num_trials_list))

# ✅ Sparse Reconstruction 실행 함수
def run_sparse_reconstruction(i, min_num_matches, min_model_size, init_num_trials):
    exp_sparse_output_path = sparse_output_path / f"sparse_{i}"
    exp_sparse_output_path.mkdir(exist_ok=True)

    print(f"\n🔍 [{i+1}] Sparse Reconstruction: min_matches={min_num_matches}, min_model_size={min_model_size}, init_trials={init_num_trials}")

    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 8
    options.ba_local_max_num_iterations = 50
    options.ba_global_max_num_iterations = 100
    options.min_num_matches = min_num_matches
    options.min_model_size = min_model_size
    options.init_num_trials = init_num_trials
    options.multiple_models = True

    try:
        reconstruction = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_dir),
            output_path=str(exp_sparse_output_path),
            options=options
        )

        num_images_registered = max([reconstruction[i].num_reg_images() for i in reconstruction])
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

# ✅ 실행 및 결과 저장
results = []
for i, params in enumerate(param_list):
    results.append(run_sparse_reconstruction(i, *params))

results_df = pd.DataFrame(results, columns=["output_path", "min_num_matches", "min_model_size", "init_num_trials", "num_images_registered", "camera_bin_size"])
sparse_results_csv = sparse_output_path / "sparse_results.csv"
results_df.to_csv(sparse_results_csv, index=False)
print("\n✅ 모든 Sparse Reconstruction 실험 완료! 결과 CSV 저장됨.")

# ✅ 이미지 백업
backup_image_dir.mkdir(exist_ok=True)
for img_file in image_dir.glob("*.png"):
    shutil.move(str(img_file), backup_image_dir / img_file.name)

# ✅ 최적 sparse 경로 선택 후 등록된 이미지 복원
best_sparse = results_df.loc[results_df["num_images_registered"].idxmax(), "output_path"]
best_sparse_path = pathlib.Path(best_sparse) / "0"

recon = pycolmap.Reconstruction(str(best_sparse_path))
registered_names = {recon.images[i].name for i in recon.images}

restored = 0
for name in registered_names:
    src = backup_image_dir / name
    dst = image_dir / name
    if src.exists():
        shutil.move(str(src), str(dst))
        restored += 1

print(f"\n📦 총 {restored}개의 등록된 이미지만 복원 완료.")
