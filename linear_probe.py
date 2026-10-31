"""Linear probe: how good are frozen SSL features for classification?

Features of the frozen encoder (no augmentation, plain resize) are extracted once for the training split,
the validation split and the test set. A linear classifier is trained on the training-split features;
its weight decay is chosen on validation, and test accuracy is reported for that choice.

    python linear_probe.py --ckpt results/ssl/simclr_resnet18.pt [--ckpt ...] --out results/linear_probe.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from cvexp import DATA_ROOT, SPLIT_SEED, VAL_FRACTION
from ssl_pretrain import MEAN, STD, build_backbone


@torch.no_grad()
def features(enc, ds, device):
    out, ys = [], []
    for x, y in DataLoader(ds, batch_size=256, num_workers=8):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(enc(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def probe(ckpt_path, device, random_init_arch=None, img_size=128):
    if random_init_arch:                                  # reference: frozen *random* encoder
        enc, _ = build_backbone(random_init_arch, img_size)
        name = f'random_{random_init_arch}'
    else:
        ck = torch.load(ckpt_path, map_location='cpu')
        enc, _ = build_backbone(ck['arch'], ck['img_size'])
        enc.load_state_dict(ck['state_dict'])
        img_size, name = ck['img_size'], ckpt_path
    torch.manual_seed(0)
    enc.to(device).eval()
    tf = transforms.Compose([transforms.Grayscale(3), transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full = datasets.ImageFolder(DATA_ROOT / 'train', transform=tf)
    val = int(round(len(full) * VAL_FRACTION))
    tr, va = random_split(full, [len(full) - val, val], generator=torch.Generator().manual_seed(SPLIT_SEED))
    xtr, ytr = features(enc, Subset(full, tr.indices), device)
    xva, yva = features(enc, Subset(full, va.indices), device)
    xte, yte = features(enc, datasets.ImageFolder(DATA_ROOT / 'test', transform=tf), device)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6                # standardise with training statistics
    xtr, xva, xte = [(x - mu) / sd for x in (xtr, xva, xte)]
    best = None
    for wd in [1e-4, 1e-3, 1e-2, 1e-1]:
        torch.manual_seed(0)
        lin = torch.nn.Linear(xtr.size(1), 16).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        a, b = xtr.to(device), ytr.to(device)
        for _ in range(500):                                # full-batch training
            opt.zero_grad(); F.cross_entropy(lin(a), b).backward(); opt.step()
        acc = lambda x, y: (lin(x.to(device)).argmax(1).cpu() == y).float().mean().item() * 100
        r = {'wd': wd, 'val': acc(xva, yva), 'test': acc(xte, yte), 'train': acc(xtr, ytr)}
        if best is None or r['val'] > best['val']:
            best = r
    return {'encoder': name, **best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='*', default=[])
    ap.add_argument('--random', nargs='*', default=[], help='architectures to probe with random init')
    ap.add_argument('--out')
    args = ap.parse_args()
    device = torch.device('cuda')
    res = [probe(None, device, random_init_arch=a) for a in args.random] + [probe(c, device) for c in args.ckpt]
    for r in res:
        print(f"{r['encoder']:45s} linear probe: val {r['val']:.1f}%  test {r['test']:.1f}%  (train {r['train']:.1f}%, wd {r['wd']:g})")
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
