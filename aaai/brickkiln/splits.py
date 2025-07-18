# brickkiln/splits.py
import os, random

def make_split_symlinks(img_dir, lbl_dir, out_train_img, out_train_lbl, out_test_img, out_test_lbl, ratio=0.6, seed=2024):
    os.makedirs(out_train_img, exist_ok=True)
    os.makedirs(out_train_lbl, exist_ok=True)
    os.makedirs(out_test_img, exist_ok=True)
    os.makedirs(out_test_lbl, exist_ok=True)
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
    random.seed(seed)
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * ratio)
    train_imgs, test_imgs = all_imgs[:n_train], all_imgs[n_train:]
    def link(img_list, dest_img_dir, dest_lbl_dir):
        for fname in img_list:
            img_src = os.path.join(img_dir, fname)
            img_dst = os.path.join(dest_img_dir, fname)
            lbl_src = os.path.join(lbl_dir, fname.replace(".tif", ".txt"))
            lbl_dst = os.path.join(dest_lbl_dir, fname.replace(".tif", ".txt"))
            if not os.path.exists(img_dst): os.symlink(img_src, img_dst)
            if os.path.exists(lbl_src) and not os.path.exists(lbl_dst): os.symlink(lbl_src, lbl_dst)
    link(train_imgs, out_train_img, out_train_lbl)
    link(test_imgs, out_test_img, out_test_lbl)
    print(f"✅ Split {img_dir}: {len(train_imgs)} train, {len(test_imgs)} test")

def make_random_symlinks(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, n=100, seed=1008, filetype=".tif", log_file=None):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    img_files = [f for f in os.listdir(src_img_dir) if f.endswith(filetype)]
    random.seed(seed)
    chosen = random.sample(img_files, n)
    if log_file:
        with open(log_file, "w") as f:
            for fname in chosen:
                f.write(fname + "\n")
    for fname in chosen:
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        src_lbl = os.path.join(src_lbl_dir, fname.replace(filetype, ".txt"))
        dst_lbl = os.path.join(dst_lbl_dir, fname.replace(filetype, ".txt"))
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl) and os.path.exists(src_lbl):
            os.symlink(src_lbl, dst_lbl)
    print(f"✅ Symlinked {n} images+labels from {src_img_dir} to {dst_img_dir}")
