#!/bin/bash

PY=/data2/home/HZNU_ZWY/anaconda3/envs/yolo/bin/python
WORKDIR=/data/HZNU_ZWY/zwy_project/ultralytics-8.1.35
LOGDIR=$WORKDIR/logs

mkdir -p "$LOGDIR"
cd "$WORKDIR" || exit 1

run_job() {
    name="$1"
    shift

    echo "=============================="
    echo "START: $name"
    echo "TIME: $(date)"
    echo "=============================="

    "$@" 2>&1 | tee "$LOGDIR/${name}.log"
    status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        echo "[FAILED] $name exited with code $status" | tee -a "$LOGDIR/${name}.log"
    else
        echo "[OK] $name finished successfully" | tee -a "$LOGDIR/${name}.log"
    fi

    echo
}


run_job coral_head \
    "$PY" train_coral_head_icdas4.py \
    --train_csv derived_338_matched_userref_balanced/icdas4_train.csv \
    --val_csv derived_338_matched_userref_balanced/icdas4_val.csv \
    --img_root_train VOCdevkit_338_matched_userref_balanced/train/images \
    --img_root_val VOCdevkit_338_matched_userref_balanced/val/images \
    --epochs 60 \
    --bs 64 \
    --lr 3e-4 \
    --expand 1.25 \
    --seed 3407 \
    --workers 4 \
    --out coral_head_icdas4_338.pt

run_job coral_strict_head \
    "$PY" train_coral_strict_head_icdas4.py \
    --train_csv derived_338_matched_userref_balanced/icdas4_train.csv \
    --val_csv derived_338_matched_userref_balanced/icdas4_val.csv \
    --img_root_train VOCdevkit_338_matched_userref_balanced/train/images \
    --img_root_val VOCdevkit_338_matched_userref_balanced/val/images \
    --epochs 60 \
    --bs 64 \
    --lr 3e-4 \
    --expand 1.25 \
    --seed 3407 \
    --workers 4 \
    --out coral_strict_head_icdas4_338.pt

run_job corn_head \
    "$PY" train_corn_head_icdas4.py \
    --train_csv derived_338_matched_userref_balanced/icdas4_train.csv \
    --val_csv derived_338_matched_userref_balanced/icdas4_val.csv \
    --img_root_train VOCdevkit_338_matched_userref_balanced/train/images \
    --img_root_val VOCdevkit_338_matched_userref_balanced/val/images \
    --epochs 60 \
    --bs 64 \
    --lr 3e-4 \
    --expand 1.25 \
    --seed 3407 \
    --workers 4 \
    --out corn_head_icdas4_338.pt

run_job softmax_head \
    "$PY" train_softmax_head_icdas4.py \
    --train_csv derived_338_matched_userref_balanced/icdas4_train.csv \
    --val_csv derived_338_matched_userref_balanced/icdas4_val.csv \
    --img_root_train VOCdevkit_338_matched_userref_balanced/train/images \
    --img_root_val VOCdevkit_338_matched_userref_balanced/val/images \
    --epochs 60 \
    --bs 64 \
    --lr 3e-4 \
    --expand 1.25 \
    --seed 3407 \
    --workers 4 \
    --out softmax_head_icdas4_338.pt \
    --head_type softmax

run_job ord2seq_head \
    "$PY" train_softmax_head_icdas4.py \
    --train_csv derived_338_matched_userref_balanced/icdas4_train.csv \
    --val_csv derived_338_matched_userref_balanced/icdas4_val.csv \
    --img_root_train VOCdevkit_338_matched_userref_balanced/train/images \
    --img_root_val VOCdevkit_338_matched_userref_balanced/val/images \
    --epochs 60 \
    --bs 64 \
    --lr 3e-4 \
    --expand 1.25 \
    --seed 3407 \
    --workers 4 \
    --out ord2seq_head_icdas4_338.pt \
    --head_type ord2seq


echo "All jobs finished. Check logs in: $LOGDIR"



