import pandas as pd
import numpy as np
import os

def process_file(inp, out):
    if not os.path.exists(inp):
        print(f"File not found: {inp}")
        return

    print(f"Processing {inp} -> {out}")
    df = pd.read_csv(inp)

    if "cls_pred" in df.columns:
        print("Already has cls_pred, just saving.")
    else:
        if "ic4" not in df.columns:
            print(f"Skipping {inp}: 'ic4' column not found. Columns: {df.columns.tolist()}")
            return

        # 允许 ic4 是整数 或 字符串标签
        if df["ic4"].dtype == object:
            m={
                "ICDAS0":0, "ICDAS_A":1, "ICDAS_B":2, "ICDAS_C":3,
                "0":0, "A":1, "B":2, "C":3
            }
            df["cls_pred"]=df["ic4"].map(m)
            if df["cls_pred"].isna().any():
                bad=df.loc[df["cls_pred"].isna(),"ic4"].unique().tolist()
                print(f"Error: ic4里出现未识别标签: {bad}")
                return
            df["cls_pred"]=df["cls_pred"].astype(int)
        else:
            df["cls_pred"]=df["ic4"].astype(int)

    df.to_csv(out, index=False)
    print("Saved:", out, "rows:", len(df))
    print("cls_pred unique:", sorted(df["cls_pred"].unique().tolist()))

if __name__ == "__main__":
    # Test set
    process_file(
        "runs/detect/twostage/pred_test_twostage_ordinal_norm.csv",
        "runs/detect/twostage/pred_test_twostage_ordinal_norm_cls.csv"
    )
    # Val set
    process_file(
        "runs/detect/twostage/pred_val_twostage_ordinal_norm.csv",
        "runs/detect/twostage/pred_val_twostage_ordinal_norm_cls.csv"
    )
