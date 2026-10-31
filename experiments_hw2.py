"""Assignment 2 experiments: supervised baselines, SSL pretraining -> fine-tuning, joint SSL + CE.

All networks are randomly initialised and trained on the provided training images only. Settings shared
by every classifier run (FT): 128x128 grayscale replicated to 3 channels, flip + random-resized crop +
translation, Adam, 200 epochs, batch 64; cosine schedule for ResNet-18, linear warm-up + cosine for
ViT-Tiny. Each configuration is trained with 3 seeds; the peak lr is chosen per method on validation
(mean of the last 10 epochs). SSL pretraining (ssl_pretrain.py) uses the 1,920 training-split images.

    python experiments_hw2.py baselines|joint|finetune|summary [--gpus 2 3]
"""
import argparse
import json

import numpy as np

from cvexp import RESULTS_DIR, ROOT, run_grid, summarize

SEEDS = [0, 1, 2]
LRS = [1e-3, 3e-4]
FT = {'img_size': 128, 'augment': ['hflip', 'rrc', 'translate'], 'epochs': 200, 'amp': True}
ARCH = {'resnet18': {'model': 'resnet18', 'schedule': 'cosine'},
        'vit_tiny': {'model': 'vit_tiny', 'schedule': 'warmup_cosine'}}
SSL_CKPTS = {   # method -> (backbone, encoder checkpoint from ssl_pretrain.py)
    'simclr': ('resnet18', 'results/ssl/simclr_resnet18_e800.pt'),
    'moco': ('resnet18', 'results/ssl/moco_resnet18_e800.pt'),
    'dino': ('resnet18', 'results/ssl/dino_resnet18_e800.pt'),
    'mae': ('vit_tiny', 'results/ssl/mae_vit_tiny_e800.pt'),
}


def configs(stage):
    """{label: config} for one stage (every config is run for every lr in LRS)."""
    if stage == 'baselines':
        return {f'supervised {a}': {**FT, **ARCH[a]} for a in ARCH}
    if stage == 'joint':      # CE + lambda * SimCLR (NT-Xent) on the same training batch (ssl_aux.py)
        return {f'joint simclr lambda={w:g} resnet18': {**FT, **ARCH['resnet18'], 'ssl': 'simclr', 'ssl_weight': w}
                for w in [0.1, 1.0]}
    if stage == 'finetune':
        return {f'{m} -> fine-tune {a}': {**FT, **ARCH[a], 'init': ck} for m, (a, ck) in SSL_CKPTS.items()}
    if stage == 'best':       # best recipe search around DINO: lower fine-tuning lr, longer pretraining
        cfgs = {'dino (800 ep) -> fine-tune resnet18': {**FT, **ARCH['resnet18'], 'init': SSL_CKPTS['dino'][1]}}
        if (ROOT / 'results/ssl/dino_resnet18_e1600.pt').exists():
            cfgs['dino (1600 ep) -> fine-tune resnet18'] = {**FT, **ARCH['resnet18'],
                                                            'init': 'results/ssl/dino_resnet18_e1600.pt'}
        return cfgs
    if stage == 'final':      # recipe selected on validation (DINO 1600 ep, lr 3e-4), now with all 2,400
        return {'final: dino (1600 ep, all images) -> fine-tune resnet18':   # training images for SSL + labels
                {**FT, **ARCH['resnet18'], 'init': 'results/ssl/dino_resnet18_e1600_all.pt', 'train_on': 'trainval'}}
    raise ValueError(stage)


STAGE_LRS = {'best': [3e-4, 1e-4, 5e-5], 'final': [3e-4]}


def run(stage, gpus, jobs_per_gpu):
    grid = [(label, lr, {**cfg, 'lr': lr}) for label, cfg in configs(stage).items() for lr in STAGE_LRS.get(stage, LRS)]
    runs = run_grid(f'hw2_{stage}', [c for _, _, c in grid], seeds=SEEDS, gpus=gpus,
                    jobs_per_gpu=jobs_per_gpu, save_checkpoint=True)      # one grid: all slots busy
    out = []
    for i, (label, lr, cfg) in enumerate(grid):
        rs = runs[i * len(SEEDS):(i + 1) * len(SEEDS)]
        h = [r['history'] for r in rs]
        l10 = np.array([np.mean(x['val_acc'][-10:]) for x in h]) * 100
        out.append(summarize(f'{label} lr={lr:g}', rs, {
            'label': label, 'lr': lr, 'config': cfg,
            'val_last10': float(l10.mean()), 'val_last10_std': float(l10.std()),
            'final_train_loss': float(np.mean([x['train_loss'][-1] for x in h])),
            'final_val_loss': float(np.mean([x['val_loss'][-1] for x in h]))}))
    json.dump(out, open(RESULTS_DIR / f'hw2_{stage}.json', 'w'), indent=2)
    return out


def test_table():
    """Per method: the lr with the best validation (last-10) mean, then test accuracy of its 3 seeds.
    The test set is not used for any choice."""
    from cvexp import config_id
    from evaluate import evaluate_checkpoint
    rows = []
    for stage in ['baselines', 'joint', 'finetune', 'best', 'final']:
        path = RESULTS_DIR / f'hw2_{stage}.json'
        if not path.exists():
            continue
        by_label = {}
        for r in json.load(open(path)):
            if r['label'] not in by_label or r['val_last10'] > by_label[r['label']]['val_last10']:
                by_label[r['label']] = r
        for label, r in by_label.items():
            accs = [evaluate_checkpoint(RESULTS_DIR / 'runs' / f'hw2_{stage}' / f"{config_id(r['config'])}_s{s}.pt",
                                        'test')['accuracy'] * 100 for s in SEEDS]
            rows.append({'stage': stage, 'label': label, 'lr': r['lr'], 'val_last10': r['val_last10'],
                         'val_last10_std': r['val_last10_std'], 'test': float(np.mean(accs)),
                         'test_std': float(np.std(accs)), 'test_seeds': accs})
            print(f"{label:42s} lr {r['lr']:<7g} val {r['val_last10']:6.2f} ± {r['val_last10_std']:4.2f}   "
                  f"test {np.mean(accs):6.2f} ± {np.std(accs):4.2f}")
    json.dump(rows, open(RESULTS_DIR / 'hw2_test_table.json', 'w'), indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['baselines', 'joint', 'finetune', 'best', 'final', 'test'])
    ap.add_argument('--gpus', nargs='+', default=None)
    ap.add_argument('--jobs-per-gpu', type=int, default=2)
    a = ap.parse_args()
    if a.stage == 'test':
        test_table()
        raise SystemExit
    for r in run(a.stage, a.gpus, a.jobs_per_gpu):
        print(f"{r['experiment']:45s} val last-10 {r['val_last10']:.2f} ± {r['val_last10_std']:.2f}  "
              f"train loss {r['final_train_loss']:.3f}  val loss {r['final_val_loss']:.3f}")
