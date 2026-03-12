# tools/calibrate_pge1_platt.py
import argparse, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def logit(p): 
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi_csv', required=True, help='roi_val_icdas4_with_probs.csv（含 p_ge1 与 icdas）')
    ap.add_argument('--out_json', required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.roi_csv)
    assert {'p_ge1','icdas'}.issubset(df.columns)
    y = (df['icdas'] >= 1).astype(int).values
    X = logit(df['p_ge1'].values).reshape(-1,1)

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, y)

    # p_cal = sigmoid(A*logit(p) + B)
    A = float(lr.coef_[0,0]); B = float(lr.intercept_[0])
    with open(a.out_json, 'w') as f:
        json.dump({'A':A, 'B':B}, f)
    print("Saved:", a.out_json, "A=%.4f B=%.4f" % (A,B))
