# 🚀 BoSS Fine-tuning & Inference Recipes

![BoSS Logo](https://yourdomain.com/assets/boss-logo.svg) <!-- Replace with your actual image -->

> 🧠 **BoSS Recipes** is the companion to the [BoSS model](https://github.com/boss-net/boss), offering hands-on recipes for fine-tuning and inference with Hugging Face support. Includes multi-GPU & PEFT examples out-of-the-box.

> 📜 For ethical considerations and limitations, please read the [Responsible Use Guide](https://github.com/boss-net/boss/blob/main/Responsible-Use-Guide.pdf).

---

## 📦 Features at a Glance

* ✅ Single & Multi-GPU Fine-tuning Recipes (PEFT, FSDP, Quantization)
* ✅ Hugging Face Compatibility
* ✅ Dataset Customization
* ✅ Efficient Inference Examples
* ✅ Slurm-based Multi-node Training
* ✅ Modular Config System

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Fine-tuning](#fine-tuning)

   * [Single GPU](#single-gpu)
   * [Multi-GPU One Node](#multiple-gpus-one-node)
   * [Multi-GPU Multi Node](#multi-gpu-multi-node)
3. [Inference](./inference/inference.md)
4. [Model Conversion](#model-conversion-to-hugging-face)
5. [Repository Organization](#repository-organization)
6. [License](#license)

---

## ⚡ Quick Start

📓 Try the [Quickstart Notebook](quickstart.ipynb) — Fine-tune BoSS on the [Samsum dataset](https://huggingface.co/datasets/samsum) using PEFT + int8 quantization on a single A10 GPU (24GB VRAM).

```bash
pip install -r requirements.txt
```

> ⚠️ Uses PyTorch 2.0.1 by default. For **FSDP + PEFT**, install [PyTorch Nightly](https://pytorch.org/get-started/locally/).

---

## 🧪 Fine-tuning

BoSS supports domain-specific adaptation using:

* **PEFT (LoRA, Prefix, BossAdapter)**
* **FSDP (Full-Shard Data Parallel)**
* **Hybrid (PEFT + FSDP)**

Explore the full [LLM Fine-tuning guide](./docs/LLM_finetuning.md) 🔍

### 🖥️ Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python boss_finetuning.py \
  --use_peft \
  --peft_method lora \
  --quantization \
  --model_name /path/to/7B \
  --output_dir /output/peft_model
```

> 💡 Ensure `save_model` is enabled in [training.py](configs/training.py)

---

### 🧠 Multi-GPU (Single Node)

> 🔧 Requires PyTorch Nightly for `FSDP + PEFT`

```bash
torchrun --nnodes 1 --nproc_per_node 4 boss_finetuning.py \
  --enable_fsdp \
  --use_peft \
  --peft_method lora \
  --pure_bf16 \
  --model_name /path/to/7B \
  --output_dir /output/peft_model
```

---

### 🔁 FSDP Only (Full Fine-tuning)

```bash
torchrun --nnodes 1 --nproc_per_node 8 boss_finetuning.py \
  --enable_fsdp \
  --model_name /path/to/7B \
  --dist_checkpoint_root_folder model_checkpoints \
  --dist_checkpoint_folder fine-tuned
```

---

### 🌐 Multi-Node (via SLURM)

```bash
sbatch multi_node.slurm
```

> ✏️ Modify node/GPU count in the SLURM script

---

## 🔄 Model Conversion to Hugging Face

To use with `transformers`, convert native BoSS weights:

```bash
pip install git+https://github.com/huggingface/transformers

python src/transformers/models/boss/convert_boss_weights_to_hf.py \
    --input_dir /path/to/original/weights \
    --model_size 7B \
    --output_dir models_hf/7B
```

---

## 🧭 Repository Structure

| Folder                 | Description                                  |
| ---------------------- | -------------------------------------------- |
| `configs/`             | Training, PEFT, FSDP configs                 |
| `docs/`                | Finetuning guides, FAQ                       |
| `ft_datasets/`         | Download & preprocess scripts                |
| `inference/`           | Inference examples                           |
| `model_checkpointing/` | FSDP checkpoint utils                        |
| `policies/`            | Precision & memory policies                  |
| `utils/`               | Helper utils (training, memory, CLI configs) |

---

## 🛡️ License & Acceptable Use

Please review the [LICENSE](LICENSE) and [USE\_POLICY.md](USE_POLICY.md) before using the code or models.

---

## 🎨 Optional: Infographic Suggestions

You could add illustrations or badges for extra visual clarity:

* **Architecture diagram** of PEFT/FSDP setup
* **Flowchart**: Model → Conversion → Training → Inference
* **Badges**:

```md
![License](https://img.shields.io/github/license/boss-net/boss-recipes)
![Stars](https://img.shields.io/github/stars/boss-net/boss-recipes?style=social)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
```
