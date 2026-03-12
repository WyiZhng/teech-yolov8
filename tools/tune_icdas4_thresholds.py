# tools/tune_icdas4_thresholds.py
import argparse, pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def pred_ic4(p1,p3,p5,t1,t3,t5):
    if p1 < t1: return 0
    if p3 < t3: return 1
    if p5 < t5: return 2
    return 3

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_csv', required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.in_csv)
    assert {'p_ge1','p_ge3','p_ge5','ic4'}.issubset(df.columns)

    best = None
    # 搜索范围可按需加密
    T1 = np.linspace(0.45, 0.70, 6)   # ≥1
    T3 = np.linspace(0.45, 0.70, 6)   # ≥3
    T5 = np.linspace(0.35, 0.60, 6)   # ≥5（更宽松，让C能叫出来）
    for t1 in T1:
        for t3 in T3:
            for t5 in T5:
                pred = [pred_ic4(r.p_ge1, r.p_ge3, r.p_ge5, t1,t3,t5) for r in df.itertuples()]
                qwk = cohen_kappa_score(df.ic4, pred, weights='quadratic')
                mae = float(np.mean(np.abs(np.array(pred) - df.ic4.values)))
                score = qwk - 0.1*mae  # 兼顾MAE，可调
                if best is None or score > best[0]:
                    best = (score, t1,t3,t5, qwk, mae, pred)

    _, t1,t3,t5, qwk, mae, pred = best
    cm = confusion_matrix(df.ic4, pred, labels=[0,1,2,3])
    print(f"Best thresholds -> t1={t1:.2f}  t3={t3:.2f}  t5={t5:.2f}")
    print(f"QWK={qwk:.3f}  MAE={mae:.3f}")
    print("Confusion Matrix (rows=GT 0/A/B/C, cols=Pred):")
    print(cm)
