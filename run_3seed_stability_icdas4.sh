#!/usr/bin/env bash
set -euo pipefail

# 避免多任务同时执行时的显存冲突，限制使用的 GPU
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="/data2/home/HZNU_ZWY/anaconda3/envs/yolo/bin/python"

# 数据配置
PROJECT_ROOT="/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35"
TRAIN_CSV="derived_338_matched_userref_balanced/icdas4_train.csv"
VAL_CSV="derived_338_matched_userref_balanced/icdas4_val.csv"
IMG_ROOT_TRAIN="VOCdevkit_338_matched_userref_balanced/train/images"
IMG_ROOT_VAL="VOCdevkit_338_matched_userref_balanced/val/images"

# 超参数
EPOCHS=60
BS=64
LR=3e-4
SEEDS=(3048 3049 3050)

# 方法与对应脚本的映射字典
declare -A METHOD_SCRIPTS
METHOD_SCRIPTS["softmax"]="train_softmax_head_icdas4.py"
METHOD_SCRIPTS["corn"]="train_corn_head_icdas4.py"
METHOD_SCRIPTS["order"]="train_order_head_icdas4.py"
METHOD_SCRIPTS["cloc"]="train_cloc_head_icdas4.py"
METHOD_SCRIPTS["BA-OGAF-Net"]="train_softmax_ordplus_order_boundary_gpt.py"

# 进入项目根目录
cd "$PROJECT_ROOT" || { echo "无法进入 $PROJECT_ROOT"; exit 1; }

# 创建输出目录
OUT_DIR="stability_3seed_runs"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "开始运行 3 Seed 稳定性实验 (Seeds: ${SEEDS[*]})"
echo "保存目录: $OUT_DIR"
echo "========================================================"

# 按顺序遍历每个方法与每个随机种子
for method in "softmax" "corn" "order" "cloc" "BA-OGAF-Net"; do
    script="${METHOD_SCRIPTS[$method]}"
    
    if [[ ! -f "$script" ]]; then
        echo "[ERROR] 未找到对应脚本 $script 对于方法 $method，跳过！"
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        # 定义输出模型与日志路径
        out_pt="${OUT_DIR}/${method}_seed${seed}.pt"
        log_file="${LOG_DIR}/${method}_seed${seed}.log"
        
        echo ""
        echo "--------------------------------------------------------"
        echo "[RUNNING] Method: $method | Seed: $seed"
        echo "Script  : $script"
        echo "Log     : $log_file"
        echo "Ckpt    : $out_pt"
        echo "--------------------------------------------------------"
        
        # 执行训练并通过 tee 写入日志
        $PYTHON_BIN "$script" \
            --train_csv "$TRAIN_CSV" \
            --val_csv "$VAL_CSV" \
            --img_root_train "$IMG_ROOT_TRAIN" \
            --img_root_val "$IMG_ROOT_VAL" \
            --epochs "$EPOCHS" \
            --bs "$BS" \
            --lr "$LR" \
            --seed "$seed" \
            --out "$out_pt" 2>&1 | tee "$log_file"
            
        echo "[DONE] Method: $method | Seed: $seed 完成。"
    done
done

echo "========================================================"
echo "所有 3 Seed 稳定性实验运行完毕！"
echo "执行结果见: $LOG_DIR"
echo "========================================================"
