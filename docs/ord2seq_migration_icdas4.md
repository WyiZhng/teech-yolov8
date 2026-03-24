# Ord2Seq Head Migration for ICDAS4

## Goal

Integrate the core Ord2Seq idea as a pluggable ordinal head for ICDAS4 (4 classes), while keeping existing data loading and train/eval loops mostly unchanged.

## Core Mechanisms Extracted from Official Ord2Seq

1. Ordinal label -> sequence label:
- Map each ordinal class to a multi-step decision sequence.
- In the official code (`make_binary_gt`), each class maps to token sequence through a predefined tree-style mapping.

2. Dichotomy tree / label sequence:
- Classes are recursively split into coarse-to-fine groups.
- Each decoding step corresponds to one level of the tree.

3. Decoder step-by-step prediction:
- Transformer decoder predicts one step at a time autoregressively.
- Training uses teacher forcing; inference feeds previous predicted token.

4. Masked decision strategy:
- Previous step narrows candidate classes.
- Current-step logits are masked by allowed candidates from previous decision.

5. Sequence -> final class:
- Last-step constrained logits are used for final class prediction.

6. Loss:
- Official implementation uses per-step BCE over class-mask targets and sums losses.
- Migrated head keeps this style with weighted sum across steps.

7. Backbone-agnostic vs task-bound:
- Backbone-agnostic: sequence head, tree construction, decoder logic, masking strategy, per-step BCE.
- Task-bound: class mapping and number of classes, output decoding to task-specific metrics.

## Implemented Files

- `ord2seq_head.py`
	- New pluggable `Ord2SeqOrdinalHead`.
	- Input: feature tensor `[B, C]`.
	- Default use case: `num_classes=4`.
	- Internal components:
		- auto-built balanced dichotomy hierarchy
		- class->sequence token paths
		- Transformer decoder
		- per-step class-mask heads
		- masked decision during autoregressive inference

- `train_softmax_head_icdas4.py`
	- Added `--head_type {softmax,ord2seq}`.
	- Existing data pipeline unchanged.
	- For `ord2seq`, the model still uses the same ResNet18 backbone and train loop structure.
	- Loss switches to head-provided Ord2Seq sequence loss.

- `tools/eval_ordinal_on_roi_icdas4.py`
	- Auto-detects checkpoint type (classic ordinal head vs Ord2Seq head).
	- Keeps current evaluation metrics/output flow.
	- For Ord2Seq, uses class probabilities to derive threshold-style scores:
		- `p_ge1 = 1 - P(class=0)`
		- `p_ge3 = P(class=2) + P(class=3)`
		- `p_ge5 = P(class=3)`

## How to Train with Ord2Seq Head

Example:

```bash
python train_softmax_head_icdas4.py \
	--train_csv icdas4_train.csv \
	--val_csv icdas4_val.csv \
	--img_root_train Benchmarking\ Dataset/train \
	--img_root_val Benchmarking\ Dataset/valid \
	--head_type ord2seq \
	--ord2seq_d_model 256 \
	--ord2seq_layers 2 \
	--out ord2seq_head_icdas4.pt
```

## How to Evaluate

```bash
python tools/eval_ordinal_on_roi_icdas4.py \
	--val_csv icdas4_val.csv \
	--img_root Benchmarking\ Dataset/valid \
	--ckpt ord2seq_head_icdas4.pt
```

