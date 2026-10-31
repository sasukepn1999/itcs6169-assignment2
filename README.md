# ITCS 6169/8169 Assignment 2: Self-Supervised Visual Representation Learning

16-class scene classification with only 2,400 labelled training images, **without any external
pretraining**: every network is randomly initialised and trained only on the provided training images.
Report: [`report.pdf`](report.pdf) (2 pages).

## Result

**Best test accuracy: 81.4 ± 0.9%** (mean ± std over 3 seeds), 6.2 points above the supervised baseline.
Final model: a ResNet-18 pretrained with **DINO** self-distillation for 1,600 epochs on the 2,400 training
images, then fine-tuned for classification on the same images.

| Method | Backbone | Linear probe (val) | Validation (last-10) | Test |
|---|---|---|---|---|
| Supervised baseline | ResNet-18 | – | 76.8 ± 1.3% | 75.2 ± 1.1% |
| Joint CE + SimCLR (λ = 0.1) | ResNet-18 | – | 78.3 ± 1.3% | 77.8 ± 0.8% |
| SimCLR → fine-tune | ResNet-18 | 61.5% | 81.9 ± 1.1% | 78.6 ± 0.5% |
| MoCo v2 → fine-tune | ResNet-18 | 72.3% | 81.6 ± 0.2% | 80.2 ± 0.7% |
| DINO (800 epochs) → fine-tune | ResNet-18 | 81.5% | 83.7 ± 0.1% | 81.3 ± 0.5% |
| Supervised baseline | ViT-Tiny | – | 60.1 ± 0.4% | 63.5 ± 0.5% |
| MAE → fine-tune | ViT-Tiny | 57.7% | 64.3 ± 0.9% | 67.8 ± 0.8% |
| DINO (1,600 epochs) → fine-tune | ResNet-18 | 81.9% | 85.5 ± 0.4% | 80.7 ± 1.0% |
| **Final: + validation images in both stages** | ResNet-18 | – | – | **81.4 ± 0.9%** |

**Protocol.** The 2,400 labelled images are split once into 1,920 training and 480 validation images
(seed 0). SSL pretraining uses only the 1,920 training images; every choice (method, learning rate,
pretraining length) is made on validation with the mean accuracy of the last 10 epochs ("last-10"). The
test set only reports the configuration selected for each method. The final model repeats the selected
recipe on all 2,400 images, so it has no validation score. "Linear probe" is a linear classifier on the
frozen SSL encoder; a frozen random ResNet-18 reaches 36.9% this way.

## Methods

* **SimCLR**: contrastive; two augmented views of an image attract, other images in the batch repel
  (NT-Xent, temperature 0.2, batch 256).
* **MoCo v2**: contrastive with a momentum (EMA) key encoder and a queue of 1,024 negatives.
* **DINO**: self-distillation; a student matches an EMA teacher across 2 global (128 px) and 6 local
  (64 px) crops; centring and sharpening prevent collapse.
* **MAE**: ViT-Tiny sees 25% of the 16×16 patches; a light decoder reconstructs the masked 75%.
* **Joint**: cross-entropy + λ · SimCLR loss during supervised training.

Grayscale images (replicated to 3 channels), 128×128. SSL views use random-resized crops, flips,
brightness/contrast jitter and blur. Fine-tuning: flip + random-resized crop + translation, Adam, 200
epochs, batch 64, cosine schedule (warm-up + cosine for ViT).

## Setup

```bash
conda create -n cv-hw2 python=3.11 -y && conda activate cv-hw2
pip install -r requirements.txt
```

Place the dataset as `data/train/<class>/*.jpg` (2,400 images) and `data/test/<class>/*.jpg` (400 images).

## Reproduce

```bash
# 1. SSL pretraining on the 1,920 training-split images (one encoder per method)
python ssl_pretrain.py --method simclr --epochs 800  --out results/ssl/simclr_resnet18_e800.pt
python ssl_pretrain.py --method moco   --epochs 800  --out results/ssl/moco_resnet18_e800.pt
python ssl_pretrain.py --method dino   --epochs 800  --out results/ssl/dino_resnet18_e800.pt
python ssl_pretrain.py --method mae    --epochs 800  --out results/ssl/mae_vit_tiny_e800.pt
python ssl_pretrain.py --method dino   --epochs 1600 --out results/ssl/dino_resnet18_e1600.pt

# 2. experiments (3 seeds, two learning rates each; runs are spread over all GPUs and cached)
python experiments_hw2.py baselines   # supervised ResNet-18 and ViT-Tiny
python experiments_hw2.py joint       # joint CE + SimCLR
python experiments_hw2.py finetune    # SSL encoder -> fine-tuning
python experiments_hw2.py best        # DINO: fine-tuning lr and pretraining length
python linear_probe.py --random resnet18 vit_tiny --ckpt results/ssl/*.pt --out results/hw2_linear_probe.json

# 3. final model: the selected recipe on all 2,400 training images (see configs/final.json)
python ssl_pretrain.py --method dino --epochs 1600 --images all --out results/ssl/dino_resnet18_e1600_all.pt
python experiments_hw2.py final
python experiments_hw2.py test        # test accuracy of every method's selected configuration

# evaluate the released final checkpoints on the test set
python evaluate.py --checkpoint results/runs/hw2_final/41ba8cb7ef_s{0,1,2}.pt
```

Seeds are fixed, and fine-tuning is deterministic for a given GPU type.

## Checkpoints

Attached to the GitHub release [v1.0](https://github.com/sasukepn1999/itcs6169-assignment2/releases/tag/v1.0)
(PyTorch `.pt`, weights + configuration):

| File | Content | Place in |
|---|---|---|
| `41ba8cb7ef_s{0,1,2}.pt` | **final model**, 3 seeds (81.4 ± 0.9% test) | `results/runs/hw2_final/` |
| `faa87847ac_s{0,1,2}.pt` | supervised ResNet-18 baseline, 3 seeds (75.2 ± 1.1% test) | `results/runs/hw2_baselines/` |
| `*_e800.pt`, `*_e1600*.pt` | SSL encoders (SimCLR, MoCo v2, DINO, MAE) | `results/ssl/` |

## Files

| File | Purpose |
|---|---|
| `ssl_pretrain.py` | SSL pretraining: SimCLR, MoCo v2, DINO (ResNet-18) and MAE (ViT-Tiny), from random init |
| `linear_probe.py` | linear classifier on frozen SSL features |
| `experiments_hw2.py` | every Assignment 2 experiment, the final model and the test table |
| `cvexp.py` | shared training library (data split, augmentation, models, schedules, fine-tuning from an SSL checkpoint, multi-GPU runner) |
| `ssl_aux.py` | joint supervised + SSL losses |
| `evaluate.py` | accuracy and per-class accuracy of checkpoints on the test set |
| `configs/final.json` | configuration of the final model |
| `results/` | SSL loss curves, per-run histories and summaries (`hw2_*.json`) |

## Environment

Python 3.11.17, PyTorch 2.14.1, torchvision 0.29.1, timm 1.0.30 (ViT architecture only, no pretrained
weights), NVIDIA H200.
