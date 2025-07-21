# ===========================
# config.py
# All experiment configurations for Brick Kiln detection domain adaptation study.
# ===========================

# ---------- PATH ROOTS ----------
base_path = "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/data"
delhi = f"{base_path}/delhi_airshed"
lucknow = f"{base_path}/lucknow_airshed_100"

# ===========================
#       DELHI-TRAINED EXPERIMENTS
# ===========================

# 1. Train on 60% Delhi, Test on Lucknow
exp1_60_train_delhi_test_lucknow = {
    "experiment_name": "exp1_60_train_delhi_test_lucknow",
    "train_images": f"{delhi}/train60/images",
    "train_labels": f"{delhi}/train60/labels_voc",
    "test_images": f"{lucknow}/images",
    "test_labels": f"{lucknow}/labels_voc"
}

# 2. Train on 60% Delhi, Test on 40% Delhi (holdout)
exp2_60_train_delhi_test_delhi40 = {
    "experiment_name": "exp2_60_train_delhi_test_delhi40",
    "train_images": f"{delhi}/train60/images",
    "train_labels": f"{delhi}/train60/labels_voc",
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

# 3. Train on 60% Delhi + 200 random Delhi NCR, Test on Lucknow
exp3_train_60_plus_200_delhi_ncr_train_lucknow_test = {
    "experiment_name": "exp3_train_60_plus_200_delhi_ncr_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{lucknow}/images",
    "test_labels": f"{lucknow}/labels_voc"
}

# 4. Train on 60% Delhi + 200 UP pool, Test on Lucknow
exp4_train_60_plus_200_uppool_train_lucknow_test = {
    "experiment_name": "exp4_train_60_plus_200_uppool_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{lucknow}/images",
    "test_labels": f"{lucknow}/labels_voc"
}

# 5. Train on 60% Delhi + 100 UP pool + 100 NCR, Test on Lucknow
exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test = {
    "experiment_name": "exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{delhi}/images_random100"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{delhi}/labels_voc_random100"
    ],
    "test_images": f"{lucknow}/images",
    "test_labels": f"{lucknow}/labels_voc"
}

# 6. Train on 60% Delhi + 100 UP pool + 100 NCR, Test on 40% Delhi
exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test = {
    "experiment_name": "exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{delhi}/images_random100"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{delhi}/labels_voc_random100"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

# 7. Train on 60% Delhi + 200 UP pool, Test on 40% Delhi
exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test = {
    "experiment_name": "exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

# 8. Train on 60% Delhi + 200 NCR, Test on 40% Delhi
exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test = {
    "experiment_name": "exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test",
    "train_images": [
        f"{delhi}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{delhi}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{delhi}/test40/images",
    "test_labels": f"{delhi}/test40/labels_voc"
}

# ===========================
#       LUCKNOW-TRAINED EXPERIMENTS
# ===========================

# 9. Train on 60% Lucknow, Test on Delhi
exp1_60_train_lucknow_test_delhi_airshed = {
    "experiment_name": "exp1_60_train_lucknow_test_delhi_airshed",
    "train_images": f"{lucknow}/train60/images",
    "train_labels": f"{lucknow}/train60/labels_voc",
    "test_images": f"{delhi}/images",
    "test_labels": f"{delhi}/labels_voc"
}

# 10. Train on 60% Lucknow, Test on 40% Lucknow
exp2_60_train_lucknow_test_lucknow_40 = {
    "experiment_name": "exp2_60_train_lucknow_test_lucknow_40",
    "train_images": f"{lucknow}/train60/images",
    "train_labels": f"{lucknow}/train60/labels_voc",
    "test_images": f"{lucknow}/test40/images",
    "test_labels": f"{lucknow}/test40/labels_voc"
}

# 11. Train on 60% Lucknow + 200 NCR, Test on Delhi
exp3_train_60_lucknow_plus_200_delhi_ncr_train_delhi_airshed_test = {
    "experiment_name": "exp3_train_60_lucknow_plus_200_delhi_ncr_train_delhi_airshed_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{delhi}/images",
    "test_labels": f"{delhi}/labels_voc"
}

# 12. Train on 60% Lucknow + 200 UP pool, Test on Delhi
exp4_train_60_lucknow_plus_200_uppool_train_delhi_test = {
    "experiment_name": "exp4_train_60_lucknow_plus_200_uppool_train_delhi_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{delhi}/images",
    "test_labels": f"{delhi}/labels_voc"
}

# 13. Train on 60% Lucknow + 100 UP pool + 100 NCR, Test on Delhi
exp5_train_60_lucknow_plus_100_uppool_plus_100_delhi_ncr_train_delhi_test = {
    "experiment_name": "exp5_train_60_lucknow_plus_100_uppool_plus_100_delhi_ncr_train_delhi_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{delhi}/images_random100"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{delhi}/labels_voc_random100"
    ],
    "test_images": f"{delhi}/images",
    "test_labels": f"{delhi}/labels_voc"
}

# 14. Train on 60% Lucknow + 100 UP pool + 100 NCR, Test on 40% Lucknow
exp6_lucknow_60_plus_100_up_pool_plus_100_delhi_ncr_train_lucknow_40_test = {
    "experiment_name": "exp6_lucknow_60_plus_100_up_pool_plus_100_delhi_ncr_train_lucknow_40_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random100",
        f"{delhi}/images_random100"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random100",
        f"{delhi}/labels_voc_random100"
    ],
    "test_images": f"{lucknow}/test40/images",
    "test_labels": f"{lucknow}/test40/labels_voc"
}

# 15. Train on 60% Lucknow + 200 UP pool, Test on 40% Lucknow
exp7_train_lucknow_60_plus_200_up_pool_train_lucknow_40_test = {
    "experiment_name": "exp7_train_lucknow_60_plus_200_up_pool_train_lucknow_40_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{base_path}/uttar_pradesh_pool_data/images_random200"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{base_path}/uttar_pradesh_pool_data/labels_voc_random200"
    ],
    "test_images": f"{lucknow}/test40/images",
    "test_labels": f"{lucknow}/test40/labels_voc"
}

# 16. Train on 60% Lucknow + 200 NCR, Test on 40% Lucknow
exp8_train_lucknow_60_plus_200_delhi_ncr_train_lucknow_40_test = {
    "experiment_name": "exp8_train_lucknow_60_plus_200_delhi_ncr_train_lucknow_40_test",
    "train_images": [
        f"{lucknow}/train60/images",
        f"{delhi}/images_random200"
    ],
    "train_labels": [
        f"{lucknow}/train60/labels_voc",
        f"{delhi}/labels_voc_random200"
    ],
    "test_images": f"{lucknow}/test40/images",
    "test_labels": f"{lucknow}/test40/labels_voc"
}

# ===========================
#       COLLECT ALL FOR EASY ITERATION
# ===========================

all_experiments = [
    # exp1_60_train_delhi_test_lucknow,
    # exp2_60_train_delhi_test_delhi40,
    # exp3_train_60_plus_200_delhi_ncr_train_lucknow_test,
    # exp4_train_60_plus_200_uppool_train_lucknow_test,
    # exp5_delhi_plus_100_uppool_plus_100_delhi_ncr_train_lucknow_test,
    # exp6_delhi_60_plus_100_up_pool_plus_100_delhi_ncr_train_delhi_40_test,
    # exp7_train_delhi_60_plus_200_up_pool_train_delhi_40_test,
    # exp8_train_delhi_60_plus_200_delhi_ncr_train_delhi_40_test,
    # exp1_60_train_lucknow_test_delhi_airshed,
    # exp2_60_train_lucknow_test_lucknow_40,
    # exp3_train_60_lucknow_plus_200_delhi_ncr_train_delhi_airshed_test,
    # exp4_train_60_lucknow_plus_200_uppool_train_delhi_test,
    # exp5_train_60_lucknow_plus_100_uppool_plus_100_delhi_ncr_train_delhi_test,
    # exp6_lucknow_60_plus_100_up_pool_plus_100_delhi_ncr_train_lucknow_40_test,
    # exp7_train_lucknow_60_plus_200_up_pool_train_lucknow_40_test,
    exp8_train_lucknow_60_plus_200_delhi_ncr_train_lucknow_40_test
]
