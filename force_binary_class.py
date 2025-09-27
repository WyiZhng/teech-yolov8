#!/usr/bin/env python3
import glob
import os

DATA_ROOT = "/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35/Benchmarking Dataset"
SUBSETS = ["train", "valid", "test"]

def process(lbl):
    changed, out = False, []
    with open(lbl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            try:
                int(float(s[0]))
                if s[0] != "0":
                    s[0] = "0"
                    changed = True
                out.append(" ".join(s))
            except (ValueError, IndexError):
                continue
    if changed:
        bak = lbl + ".bak"
        if not os.path.exists(bak):
            os.rename(lbl, bak)
        with open(lbl, "w", encoding="utf-8") as f:
            for l in out:
                f.write(l + "\n")
    return changed

def main():
    cnt = 0
    for split in SUBSETS:
        d = os.path.join(DATA_ROOT, split, "labels")
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.txt")):
            if process(p):
                cnt += 1
                print("[remap->0]", p)
    print("done, files changed:", cnt)

if __name__ == "__main__":
    main()
