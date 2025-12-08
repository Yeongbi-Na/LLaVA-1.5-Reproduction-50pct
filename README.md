
# 🧪 LLaVA-1.5 Reproduction: Data Efficiency Study
> **Deep Learning Course Project**
>
> **Topic:** Verifying the Data Efficiency of LLaVA-1.5 with 50% Instruction Tuning Data

## 📖 Project Overview
This project reproduces the **LLaVA-1.5 7B** model to verify its "Data Efficiency" claim. We performed full fine-tuning using only **50% of the instruction dataset (approx. 332K samples)** and evaluated the model on key benchmarks (MMBench, ScienceQA, POPE).

* **Base Model:** Vicuna-7B v1.5
* **Vision Tower:** CLIP-ViT-L-336px
* **Dataset:** 50% random subset of `llava_v1_5_mix665k.json`
* **Hardware:** 4x NVIDIA A100 (80GB)

---

## 📊 Conclusion & Analysis

### 1. Conclusion
* **Verified Data Efficiency:** Successfully reproduced performance with **50% data (332K)**, confirming the paper's claim that the model is "surprisingly data-efficient".
* **Quality Over Quantity:** Demonstrated that data quality takes precedence over quantity in Visual Instruction Tuning.

### 2. Limitations
* **Random Sampling Bias:** Results may rely on the specific random split. Future work should explore **Active Data Selection** strategies.
* **Evaluation Constraints:** Verified MMBench performance using the **Dev Set** due to submission limits on the official server.

---

## 🛠️ Usage

### 1. Data Preparation
The dataset must be prepared separately. Please refer to the [official LLaVA-1.5 repository](https://github.com/haotian-liu/LLaVA) (now renamed to `README_official.md` in this repo) to download the `llava_v1_5_mixture665.json` and the corresponding image datasets.

**Important Note:** LLaVA-1.5 supports **JPG format only**. Before training, please refer to `convert_data_format.ipynb` to convert any non-JPG images (e.g., GIFs, PNGs) in the dataset to JPG format.

### 2. Evaluation Scripts
We verify data efficiency using **MMBench**, **ScienceQA**, and **POPE**.

> **Note:** The scripts below are configured for our specific NAS environment. Please update the `MODEL_PATH`, `DATA_DIR`, and `OUTPUT_DIR` variables to match your local setup.

#### 2.1 MMBench (Dev Set)
Evaluation on MMBench Dev set. Images are decoded from base64 in the TSV file.

```bash
#!/bin/bash
# Path Configuration
export MODEL_PATH="/nas/home/ongv1109/LLaVA1.5/checkpoints/llava-v1.5-7b-official"
export DATA_DIR="/nas/datahub/llava-v1.5-instruct/eval/mmbench"
export OUTPUT_DIR="./playground/data/eval/mmbench/answers/official"
export SPLIT="mmbench_dev_20230712" 

export CUDA_VISIBLE_DEVICES=4
mkdir -p $OUTPUT_DIR

echo "Step 1: Inference..."
python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file $DATA_DIR/$SPLIT.tsv \
    --answers-file $OUTPUT_DIR/$SPLIT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "Step 2: Evaluation..."
python scripts/convert_mmbench_for_submission.py \
    --annotation-file $DATA_DIR/$SPLIT.tsv \
    --result-dir $OUTPUT_DIR \
    --upload-dir $OUTPUT_DIR \
    --experiment $SPLIT
````

#### 2.2 ScienceQA

Evaluation on ScienceQA test set. Requires `problems.json` and `pid_splits.json`.

```bash
#!/bin/bash
export GPU_ID=4
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Path Configuration
export MODEL_PATH="/nas/home/ongv1109/LLaVA1.5/checkpoints/llava-v1.5-7b-official"
export DATA_DIR="/nas/datahub/llava-v1.5-instruct/eval/scienceqa"
export IMAGE_PATH="/nas/datahub/ScienceQA/test" 
export QUESTION_FILE="$DATA_DIR/llava_test_CQM-A.json"
export OUTPUT_DIR="./playground/data/eval/scienceqa/answers/official"

mkdir -p $OUTPUT_DIR
echo "Running ScienceQA evaluation on GPU $GPU_ID..."

# Step 1: Inference
python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_PATH \
    --answers-file $OUTPUT_DIR/llava_sqa_results.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "Step 2: Evaluation..."
python llava/eval/eval_science_qa.py \
    --base-dir $DATA_DIR \
    --result-file $OUTPUT_DIR/llava_sqa_results.jsonl \
    --output-file $OUTPUT_DIR/final_score.json \
    --output-result $OUTPUT_DIR/final_score_detail.json
```

#### 2.3 POPE

Evaluation on object hallucination using COCO Val 2014 images.

```bash
#!/bin/bash
export GPU_ID=5
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Path Configuration
export MODEL_PATH="/nas/home/ongv1109/LLaVA1.5/checkpoints/llava-v1.5-7b-official"
export DATA_DIR="/nas/datahub/llava-v1.5-instruct/eval/pope"
export IMAGE_DIR="/nas/datahub/llava-v1.5-instruct/coco/val2014/val2014"
export ANSWERS_DIR="./playground/data/eval/pope/answers/official"
export QUESTION_FILE="$DATA_DIR/llava_pope_test.jsonl"

mkdir -p $ANSWERS_DIR
echo "Running POPE evaluation on GPU $GPU_ID..."

# Step 1: Inference
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_DIR \
    --answers-file $ANSWERS_DIR/llava_pope_test.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "Step 2: Evaluation..."
python llava/eval/eval_pope.py \
    --annotation-dir $DATA_DIR/answers \
    --question-file $QUESTION_FILE \
    --result-file $ANSWERS_DIR/llava_pope_test.jsonl
```

-----

## Acknowledgement

This repository is based on the [Official LLaVA Repository](https://github.com/haotian-liu/LLaVA).

```
```
