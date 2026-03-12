import os
import glob

splits = ['train', 'val', 'test']
root_new = 'datasets/icdas_yolo_icdas4'
root_old = 'datasets/icdas_yolo'

for split in splits:
    dir_new = os.path.join(root_new, split, 'images')
    dir_old = os.path.join(root_old, split, 'images')
    
    # Check if it is a directory symlink
    if os.path.islink(dir_new):
        print(f"Removing directory symlink {dir_new}")
        os.unlink(dir_new)
    elif os.path.isdir(dir_new):
        # If it's a real directory, check if it's empty or has symlinks
        # We assume if it's not a link, we might need to clean it or it's already good?
        # But wait, if I just created it as a symlink in previous step, it should be a link.
        pass
    
    if not os.path.exists(dir_new):
        os.makedirs(dir_new)
        
    # Symlink files
    files = glob.glob(os.path.join(dir_old, '*'))
    print(f"Symlinking {len(files)} files from {dir_old} to {dir_new}")
    for src in files:
        src_abs = os.path.abspath(src)
        dst = os.path.join(dir_new, os.path.basename(src))
        if os.path.islink(dst):
            os.unlink(dst)
        if not os.path.exists(dst):
            os.symlink(src_abs, dst)
