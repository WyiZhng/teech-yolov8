#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
import numpy as np

LOG_DIR = "stability_3seed_runs/logs"
OUT_CSV = "stability_3seed_runs/stability_summary.csv"
OUT_MD = "stability_3seed_runs/stability_summary.md"

def extract_metrics(log_path):
    """
    从日志文件中提取 val MAE / QWK，由于不同脚本可能保存最好指标的方式不同，
    这里将扫描日志中所有的验证指标打分，并记录该次运行过程中最好的 QWK 以及对应的 MAE。
    """
    # 尽可能兼容大小写
    # Regex 1: 提取 val_mae 或 val_MAE
    mae_pattern = re.compile(r'val_[mM][aA][eE]=([0-9.]+)')
    qwk_pattern = re.compile(r'val_[qQ][wW][kK]=([0-9.]+)')
    
    # 也可以单独提取 test MAE 和 QWK 如果记录了的话
    test_mae_pattern = re.compile(r'test_[mM][aA][eE]=([0-9.]+)')
    test_qwk_pattern = re.compile(r'test_[qQ][wW][kK]=([0-9.]+)')

    val_maes, val_qwks = [], []
    test_maes, test_qwks = [], []

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # val
                mae_match = mae_pattern.search(line)
                qwk_match = qwk_pattern.search(line)
                if mae_match:
                    val_maes.append(float(mae_match.group(1)))
                if qwk_match:
                    val_qwks.append(float(qwk_match.group(1)))
                
                # test
                t_mae_match = test_mae_pattern.search(line)
                t_qwk_match = test_qwk_pattern.search(line)
                if t_mae_match:
                    test_maes.append(float(t_mae_match.group(1)))
                if t_qwk_match:
                    test_qwks.append(float(t_qwk_match.group(1)))
                    
    except Exception as e:
        print(f"Error reading {log_path}: {e}")

    # 如果日志中有明确的 "best_qwk=...", 我们尝试直接捕获最后一次
    best_qwk_pattern = re.compile(r'best_[qQ][wW][kK]=([0-9.]+)')
    best_mae_pattern = re.compile(r'best_[mM][aA][eE]=([0-9.]+)')
    
    best_qwk, best_mae = None, None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            best_qwks_matches = best_qwk_pattern.findall(content)
            if best_qwks_matches:
                best_qwk = float(best_qwks_matches[-1])  # 最后一个最好
                
            best_maes_matches = best_mae_pattern.findall(content)
            if best_maes_matches:
                best_mae = float(best_maes_matches[-1])
    except Exception:
        pass

    # 如果没有特定的 best 打印，就通过遍历的列表推导，一般以 QWK 最高为模型选中标准
    if best_qwk is None and len(val_qwks) > 0:
        best_qwk = max(val_qwks)
        # 根据 best_qwk 所在的 epoch 索引寻找对应的 mae，假设日志中的对应关系对齐
        best_idx = val_qwks.index(best_qwk)
        if best_idx < len(val_maes):
            best_mae = val_maes[best_idx]
            
    if best_mae is None and len(val_maes) > 0:
        best_mae = min(val_maes)
        
    res = {
        'val_mae': best_mae if best_mae is not None else float('nan'),
        'val_qwk': best_qwk if best_qwk is not None else float('nan')
    }
    
    # 可以附加 test 结果
    if len(test_qwks) > 0:
        res['test_qwk'] = test_qwks[-1]
    if len(test_maes) > 0:
        res['test_mae'] = test_maes[-1]

    return res

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Directory {LOG_DIR} does not exist.")
        return

    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    # regex 用于在文件名中提取 {method}_seed{seed}.log
    filename_pattern = re.compile(r'(.+)_seed(\d+)\.log')

    records = []
    
    for log_path in sorted(log_files):
        basename = os.path.basename(log_path)
        match = filename_pattern.search(basename)
        if match:
            method, seed = match.group(1), match.group(2)
            metrics = extract_metrics(log_path)
            
            records.append({
                'Method': method,
                'Seed': seed,
                'Val_MAE': metrics.get('val_mae', float('nan')),
                'Val_QWK': metrics.get('val_qwk', float('nan'))
            })
            
            # 如果有 test 信息
            if 'test_mae' in metrics:
                records[-1]['Test_MAE'] = metrics['test_mae']
            if 'test_qwk' in metrics:
                records[-1]['Test_QWK'] = metrics['test_qwk']
        else:
            print(f"Skipping unrecognized filename format: {basename}")

    if not records:
        print("No valid logs parsed from directory.")
        return

    df = pd.DataFrame(records)
    print("Parsed Data:")
    print(df)
    
    # 根据 Method 聚合并计算 Mean / Std
    df_agg = df.groupby('Method').agg(
        mae_mean=('Val_MAE', 'mean'),
        mae_std=('Val_MAE', 'std'),
        qwk_mean=('Val_QWK', 'mean'),
        qwk_std=('Val_QWK', 'std')
    ).reset_index()
    
    # 格式化输出字符串 mean ± std
    df_agg['Val_MAE_Summary'] = df_agg.apply(lambda row: f"{row['mae_mean']:.4f} ± {row['mae_std']:.4f}" if pd.notnull(row['mae_std']) else f"{row['mae_mean']:.4f}", axis=1)
    df_agg['Val_QWK_Summary'] = df_agg.apply(lambda row: f"{row['qwk_mean']:.4f} ± {row['qwk_std']:.4f}" if pd.notnull(row['qwk_std']) else f"{row['qwk_mean']:.4f}", axis=1)

    print("\nSummarized Results:")
    print(df_agg[['Method', 'mae_mean', 'mae_std', 'qwk_mean', 'qwk_std']])
    
    # 生成 CSV 和 Markdown
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved raw data to {OUT_CSV}")
    
    md_table = df_agg[['Method', 'mae_mean', 'mae_std', 'qwk_mean', 'qwk_std']].to_markdown(index=False)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# 3-Seed Stability Summary\n\n")
        f.write("## Validation Metrics (Mean ± Std)\n\n")
        f.write(md_table)
        f.write("\n")
        
    print(f"Saved markdown summary to {OUT_MD}")

if __name__ == '__main__':
    main()