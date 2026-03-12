import json, argparse, numpy as np, pandas as pd

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def logit(p):   return np.log(np.clip(p,1e-6,1-1e-6)) - np.log(1-np.clip(p,1e-6,1-1e-6))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--calib_json", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tau", type=float, default=0.60)
    ap.add_argument("--lam", type=float, default=0.40)
    ap.add_argument("--mu",  type=float, default=0.15)
    ap.add_argument("--gate_sev", action="store_true", help="仅当 p_ge1_cal ≥ tau 时才应用严重度加分")
    a = ap.parse_args()

    df = pd.read_csv(a.in_csv)

    # 需要的列：yolo_score_raw, p_ge1, p_ge3, p_ge5
    for col in ["yolo_score_raw", "p_ge1", "p_ge3", "p_ge5"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    # 1) 读取或计算 p_ge1_cal
    if a.calib_json:
        with open(a.calib_json, "r") as f:
            js = json.load(f)
        A, B = js["A"], js["B"]  # Platt: p_cal = sigmoid(A*logit(p) + B)
        p = np.clip(df["p_ge1"].values, 1e-6, 1-1e-6)
        df["p_ge1_cal"] = sigmoid(A * logit(p) + B)
    elif "p_ge1_cal" not in df.columns:
        df["p_ge1_cal"] = df["p_ge1"]

    # 2) 严重度（未校准即可）：sev ∈ [0,1]
    df["sev"] = (df["p_ge1"] + df["p_ge3"] + df["p_ge5"]) / 3.0

    # 3) 计算 Bump（双门槛：严重度只在 p_ge1_cal ≥ tau 才启用）
    p_cal = df["p_ge1_cal"].values
    sev   = df["sev"].values
    yraw  = df["yolo_score_raw"].values

    bonus_conf = a.lam * np.clip(p_cal - a.tau, 0.0, None)
    if a.gate_sev:
        mask = (p_cal >= a.tau).astype(float)
        bonus_sev = a.mu * sev * mask
    else:
        bonus_sev = a.mu * sev

    df["score_bump"] = yraw * (1.0 + bonus_conf) * (1.0 + bonus_sev)

    df.to_csv(a.out_csv, index=False)
    print(f"Wrote: {a.out_csv} | tau={a.tau} lam={a.lam} mu={a.mu} gate_sev={a.gate_sev}")
