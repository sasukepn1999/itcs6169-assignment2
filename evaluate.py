"""Evaluate saved checkpoint(s) on the test set (or on the validation split).

    python evaluate.py --checkpoint results/runs/final/<id>_s0.pt [more.pt ...] [--split test|val]

Each checkpoint stores its training config, so the model and the (non-augmented) evaluation
preprocessing are rebuilt exactly as used for validation during training. With several
checkpoints (e.g. 3 seeds) the mean ± std accuracy is reported as well.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from cvexp import DATA_ROOT, SPLIT_SEED, VAL_FRACTION, build_model, build_transform, full_config


@torch.inference_mode()
def evaluate_checkpoint(path, split='test', batch_size=64, num_workers=4):
    ckpt = torch.load(path, map_location='cpu')
    cfg = full_config({k: v for k, v in ckpt['config'].items()})
    tf = build_transform(cfg, train=False)
    if split == 'test':
        ds = datasets.ImageFolder(DATA_ROOT / 'test', transform=tf)
        classes, targets = ds.classes, ds.targets
    else:
        full = datasets.ImageFolder(DATA_ROOT / 'train', transform=tf)
        val_size = int(round(len(full) * VAL_FRACTION))
        tr, va = random_split(full, [len(full) - val_size, val_size],
                              generator=torch.Generator().manual_seed(SPLIT_SEED))
        idx = va.indices if split == 'val' else tr.indices
        ds, classes = Subset(full, idx), full.classes
        targets = [full.targets[i] for i in idx]
    assert classes == ckpt['classes'], 'class order differs from the checkpoint'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model({**cfg, 'pretrained': False}, len(classes))   # weights come from the checkpoint
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    preds = []
    for x, _ in DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp'] and device.type == 'cuda'):
            preds.append(model(x.to(device)).float().argmax(1).cpu())
    preds, targets = torch.cat(preds).numpy(), np.array(targets)
    per_class = {c: float((preds[targets == i] == i).mean()) for i, c in enumerate(classes)}
    return {'checkpoint': str(path), 'split': split, 'seed': ckpt['seed'], 'config': cfg,
            'accuracy': float((preds == targets).mean()), 'per_class': per_class,
            'preds': preds.tolist(), 'targets': targets.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--out', help='optional JSON file for the results')
    args = parser.parse_args()
    if args.split == 'test':
        print('NOTE: evaluating on the TEST set. Use it only for the final, already-selected model.')
    results = [evaluate_checkpoint(p, args.split) for p in args.checkpoint]
    for r in results:
        print(f"{Path(r['checkpoint']).name}: {args.split} accuracy {r['accuracy'] * 100:.2f}%")
    accs = np.array([r['accuracy'] for r in results]) * 100
    if len(results) > 1:
        print(f'mean ± std over {len(results)} checkpoints: {accs.mean():.2f} ± {accs.std():.2f}%')
    per_class = {c: np.mean([r['per_class'][c] for r in results]) * 100 for c in results[0]['per_class']}
    print('per-class accuracy (mean over checkpoints):')
    for c, a in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f'  {c:13s} {a:5.1f}%')
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
