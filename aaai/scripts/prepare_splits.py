# scripts/prepare_splits.py
from brickkiln.splits import make_split_symlinks, make_random_symlinks

base_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data"
lucknow = f"{base_path}/lucknow_airshed_100"
delhi   = f"{base_path}/delhi_airshed"

# --- 1. Lucknow 60:40 split ---
make_split_symlinks(
    img_dir=f"{lucknow}/images",
    lbl_dir=f"{lucknow}/labels_voc",
    out_train_img=f"{lucknow}/train60/images",
    out_train_lbl=f"{lucknow}/train60/labels_voc",
    out_test_img=f"{lucknow}/test40/images",
    out_test_lbl=f"{lucknow}/test40/labels_voc",
    ratio=0.6, seed=2024
)

# --- 2. Delhi 60:40 split ---
make_split_symlinks(
    img_dir=f"{delhi}/images",
    lbl_dir=f"{delhi}/labels_voc",
    out_train_img=f"{delhi}/train60/images",
    out_train_lbl=f"{delhi}/train60/labels_voc",
    out_test_img=f"{delhi}/test40/images",
    out_test_lbl=f"{delhi}/test40/labels_voc",
    ratio=0.6, seed=2024
)

# --- 3. Random splits for experiments ---
make_random_symlinks(
    src_img_dir=f"{base_path}/test_delhi_airshed/images",
    src_lbl_dir=f"{base_path}/test_delhi_airshed/labels_voc",
    dst_img_dir=f"{delhi}/images_random200",
    dst_lbl_dir=f"{delhi}/labels_voc_random200",
    n=200, seed=2024, log_file="random200_testdelhi.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/uttar_pradesh_pool_data/images",
    src_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc",
    dst_img_dir=f"{base_path}/uttar_pradesh_pool_data/images_random200",
    dst_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200",
    n=200, seed=2024, log_file="random200_uppool.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/uttar_pradesh_pool_data/images",
    src_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc",
    dst_img_dir=f"{base_path}/uttar_pradesh_pool_data/images_random100",
    dst_lbl_dir=f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
    n=100, seed=2024, log_file="random100_uppool.txt"
)
make_random_symlinks(
    src_img_dir=f"{base_path}/test_delhi_airshed/images",
    src_lbl_dir=f"{base_path}/test_delhi_airshed/labels_voc",
    dst_img_dir=f"{delhi}/images_random100",
    dst_lbl_dir=f"{delhi}/labels_voc_random100",
    n=100, seed=2024, log_file="random100_testdelhi.txt"
)

print("\n✅ All splits and symlinks ready.")
