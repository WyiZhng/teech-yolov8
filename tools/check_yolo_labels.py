import glob
import os
import argparse
from tqdm import tqdm

def check_labels(root_dir, nc):
    print(f"Checking labels in {root_dir} with nc={nc}...")
    label_files = glob.glob(os.path.join(root_dir, '**', '*.txt'), recursive=True)
    
    errors = 0
    for f in tqdm(label_files):
        with open(f, 'r') as rf:
            for ln, line in enumerate(rf, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Check 1: 5 parts
                if len(parts) != 5:
                    print(f'❌ Format error (not 5 columns): {f} line {ln} -> {line.strip()}')
                    errors += 1
                    continue

                # Check 2: Class ID
                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    print(f'❌ Parse error (class not int): {f} line {ln} -> {line.strip()}')
                    errors += 1
                    continue
                
                if cls < 0 or cls >= nc:
                    print(f'❌ Class ID out of bounds (nc={nc}): {f} line {ln} cls={cls}')
                    errors += 1
                
                # Check 3: BBox values
                try:
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    print(f'❌ Parse error (bbox not float): {f} line {ln} -> {line.strip()}')
                    errors += 1
                    continue

                if w <= 0 or h <= 0:
                    print(f'❌ Invalid size (w/h <= 0): {f} line {ln} w={w}, h={h}')
                    errors += 1
                
                if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                     # Just a warning, technically allowed but suspicious if far out
                    if cx < -0.5 or cx > 1.5 or cy < -0.5 or cy > 1.5:
                        print(f'⚠️ Suspicious center: {f} line {ln} cx={cx}, cy={cy}')

                if w > 1 or h > 1:
                     # Just a warning
                     if w > 1.5 or h > 1.5:
                        print(f'⚠️ Suspicious size: {f} line {ln} w={w}, h={h}')

    if errors == 0:
        print("✅ All labels passed checks.")
    else:
        print(f"❌ Found {errors} errors.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/icdas_yolo_icdas4')
    parser.add_argument('--nc', type=int, default=4)
    args = parser.parse_args()
    
    check_labels(args.root, args.nc)
