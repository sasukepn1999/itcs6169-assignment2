"""Experiment runner for Assignment 1: one (config, seed) per process, spread over all GPUs.

The training loop reproduces the starter notebook's `train_model` exactly (same split,
seeding, loss, best-validation checkpoint selection), so results are comparable with the
notebook baseline; e.g. TNet @ 64x64, seed 0 gives 49.79% validation accuracy in both.

Notebook usage:
    from cvexp import run_grid, summarize
    runs = run_grid('exp1_resolution', [{'img_size': 256}, ...])

CLI (one job, used internally by run_grid):
    python cvexp.py --config '{"img_size": 256}' --seed 0 --out results/runs/x.json
"""
import argparse
import copy
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'

DEFAULTS = {
    'model': 'tnet',          # 'tnet' (starter TNet, classifier sized to the input)
    'img_size': 64,
    'resize': 'squash',       # 'squash' = Resize((s, s)) as in the starter
    'grayscale': True,
    'optimizer': 'adam',
    'lr': 0.002,
    'weight_decay': 0.0,
    'epochs': 20,
    'batch_size': 64,
    'augment': [],            # training-only augmentations, any subset of AUGMENTATIONS
    'schedule': 'constant',   # learning-rate schedule, one of SCHEDULES (lr = peak lr)
    'warmup_epochs': 5,       # only used by 'warmup_cosine'
    'amp': False,             # bf16 autocast for forward/loss (used for the deep CNNs)
    'keep_color': False,      # keep originally-RGB images in colour (others stay grayscale)
    'ssl': 'none',            # self-supervised auxiliary task on the training images (ssl_aux.py)
    'ssl_weight': 0.0,        # loss = CE + ssl_weight * SSL loss
    'ssl_source': 'train',    # images for the SSL loss: 'train', 'val', 'test' or 'test2' (labels never read)
    'train_on': 'train',      # labelled data: 'train' (1,920-image split) or 'trainval' (all 2,400)
    'fixmatch_tau': 0.95,     # FixMatch confidence threshold for pseudo-labels (ssl='fixmatch')
    'pretrained': False,      # ImageNet-1k weights (torchvision IMAGENET1K_V1), all layers fine-tuned
    'init': '',               # path to an SSL encoder checkpoint (ssl_pretrain.py) to fine-tune from
}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Standard CNNs from torchvision, built with random init (weights=None) and a 16-way head.
# Grayscale input is replicated to 3 channels so a later pretrained run changes only the init.
TORCHVISION_MODELS = ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'convnext_tiny',
                      'mobilenet_v3_large', 'efficientnet_b0']
TIMM_MODELS = {'vit_tiny': 'vit_tiny_patch16_224'}   # architecture only, always randomly initialised
# Learning-rate schedules, applied per iteration as a multiplier of the peak lr.
# t = fraction of training completed (0 -> 1).
SCHEDULES = {
    'constant': 'lr throughout (starter behaviour)',
    'step': 'x0.1 at t=0.5 and again at t=0.75',
    'exp': 'exponential decay to 1% of lr at t=1',
    'cosine': 'half-cosine from lr to 0',
    'warmup_cosine': 'linear warm-up from 0 over warmup_epochs, then cosine to 0',
    'onecycle': 'OneCycleLR: cosine warm-up from lr/25 over 30%, then cosine to lr/25e4',
}
# Conventional augmentations (training set only; validation always uses the plain resize).
AUGMENTATIONS = {
    'hflip': 'RandomHorizontalFlip(p=0.5)',
    'translate': 'RandomCrop(s, padding=s//8, padding_mode=reflect) after the resize (8 px at 64)',
    'rrc': 'RandomResizedCrop(s, scale=(0.5, 1.0)) instead of the resize',
    'rotate': 'RandomRotation(10)',
    'jitter': 'ColorJitter(brightness=0.3, contrast=0.3)',
    'erase': 'RandomErasing(p=0.5) on the normalised tensor',
}
# Fixed so that augmented runs are reproducible: each loader worker gets its own RNG
# stream, so the augmentation draws depend on the number of workers.
NUM_WORKERS = 4
SPLIT_SEED, VAL_FRACTION = 0, 0.20


# ----------------------------------------------------------------------------- training
def set_random_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, num_classes):
    model = _build_model(cfg, num_classes)
    if cfg['init']:
        load_ssl_encoder(model, cfg)
    return model


def load_ssl_encoder(model, cfg):
    """Copy an SSL-pretrained encoder into the classifier; only the classification head stays new."""
    import torch
    ck = torch.load(ROOT / cfg['init'], map_location='cpu')
    assert ck['arch'] == cfg['model'] and ck['img_size'] == cfg['img_size'], (ck['arch'], ck['img_size'])
    missing, unexpected = model.load_state_dict(ck['state_dict'], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith(('fc.', 'head.')) for k in missing), missing


def _build_model(cfg, num_classes):
    import torch.nn as nn
    if cfg['model'] in TIMM_MODELS:
        import timm
        assert not cfg['pretrained'], 'no external pretraining in Assignment 2'
        return timm.create_model(TIMM_MODELS[cfg['model']], pretrained=False, img_size=cfg['img_size'],
                                 num_classes=num_classes)

    if cfg['model'] == 'tnet':
        side = (cfg['img_size'] - 2) // 4      # 3x3 conv (no padding) then 4x4 pool
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(in_channels(cfg), 16, kernel_size=3),
                          nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=4, stride=4)),
            nn.Sequential(nn.Flatten(), nn.Linear(16 * side * side, num_classes)),
        )
    if cfg['model'] in TORCHVISION_MODELS:
        import torchvision
        if not cfg['pretrained']:
            return getattr(torchvision.models, cfg['model'])(weights=None, num_classes=num_classes)
        # ImageNet-1k backbone; only the final linear layer is replaced (random init, 16 classes).
        model = getattr(torchvision.models, cfg['model'])(weights='IMAGENET1K_V1')
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"unknown model {cfg['model']}")


class UnlabeledImages:
    """All images under a folder (recursively), returned with a dummy label of 0.

    Used for SSL on held-out images: 'test' (class sub-folders, labels ignored) and 'test2'
    (flat folder, no labels).
    """
    def __init__(self, root, transform, classes=None):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        self.transform = transform
        # Diagnostic-only labels (from the class sub-folder, -1 if none). They are NEVER used
        # in a loss; FixMatch uses them only to log how accurate its pseudo-labels are.
        self.diag_labels = [classes.index(p.parent.name) if classes and p.parent.name in classes else -1
                            for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as img:
            return self.transform(img.copy()), i          # index, not a label


class WeakStrong:
    """FixMatch views of one unlabelled image: (weak, strong)."""
    def __init__(self, weak, strong):
        self.weak, self.strong = weak, strong

    def __call__(self, img):
        return self.weak(img), self.strong(img)


def build_strong_transform(cfg):
    """FixMatch strong augmentation: the weak (training) pipeline + RandAugment(2, 10) before
    ToTensor, + Cutout (RandomErasing, p=1) after normalisation."""
    from torchvision import transforms
    weak = build_transform(cfg, train=True).transforms
    assert isinstance(weak[-2], transforms.ToTensor) and isinstance(weak[-1], transforms.Normalize), weak
    return transforms.Compose(weak[:-2] + [transforms.RandAugment(num_ops=2, magnitude=10)] + weak[-2:] +
                              [transforms.RandomErasing(p=1.0, scale=(0.02, 0.25), ratio=(1.0, 1.0), value=0)])


class ToRGB:
    """Grayscale ('L') images -> 3 identical channels; RGB images keep their colour."""
    def __call__(self, img):
        return img.convert('RGB')

    def __repr__(self):
        return 'ToRGB()'


def in_channels(cfg):
    """TNet keeps the starter's 1-channel input; torchvision CNNs take 3 channels."""
    return 1 if cfg['grayscale'] and cfg['model'] == 'tnet' else 3


def build_transform(cfg, train):
    """Preprocessing for one split; augmentations are applied only when `train` is True."""
    from torchvision import transforms

    s = cfg['img_size']
    aug = set(cfg['augment']) if train else set()
    c = in_channels(cfg)
    if cfg['keep_color']:
        if c != 3:
            raise ValueError('keep_color needs a 3-channel model')
        tf = [ToRGB()]
    else:
        tf = [transforms.Grayscale(num_output_channels=c)] if cfg['grayscale'] else [ToRGB()]
    if cfg['resize'] != 'squash':
        raise ValueError(f"unknown resize {cfg['resize']}")
    if 'rrc' in aug:
        tf.append(transforms.RandomResizedCrop(s, scale=(0.5, 1.0)))
    else:
        tf.append(transforms.Resize((s, s)))
    if 'translate' in aug:
        # Padding scales with the input (s // 8 = 8 px at 64x64), so the relative shift is the
        # same at every resolution.
        tf.append(transforms.RandomCrop(s, padding=s // 8, padding_mode='reflect'))
    if 'hflip' in aug:
        tf.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'rotate' in aug:
        tf.append(transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR))
    if 'jitter' in aug:
        tf.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
    if cfg['pretrained']:    # the statistics the ImageNet weights were trained with
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        norm = transforms.Normalize(mean=[0.5] * c, std=[0.5] * c)
    tf += [transforms.ToTensor(), norm]
    if 'erase' in aug:
        tf.append(transforms.RandomErasing(p=0.5))
    return transforms.Compose(tf)


def build_scheduler(cfg, optimizer, steps_per_epoch):
    """Per-iteration lr scheduler; None for 'constant' so the starter loop is untouched."""
    import math
    import torch.optim.lr_scheduler as sched

    name, total = cfg['schedule'], cfg['epochs'] * steps_per_epoch
    if name == 'constant':
        return None
    if name == 'onecycle':
        return sched.OneCycleLR(optimizer, max_lr=cfg['lr'], total_steps=total)
    warmup = cfg['warmup_epochs'] * steps_per_epoch
    factors = {
        'step': lambda i: 0.1 ** ((i >= 0.5 * total) + (i >= 0.75 * total)),
        'exp': lambda i: 0.01 ** (i / total),
        'cosine': lambda i: 0.5 * (1 + math.cos(math.pi * i / total)),
        'warmup_cosine': lambda i: ((i + 1) / warmup if i < warmup else
                                    0.5 * (1 + math.cos(math.pi * (i - warmup) / (total - warmup)))),
    }
    if name not in factors:
        raise ValueError(f'unknown schedule {name}')
    return sched.LambdaLR(optimizer, factors[name])


def train_one(cfg, seed, num_workers=NUM_WORKERS):
    """Train one configuration with one seed; mirrors the notebook's `train_model`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision import datasets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from ssl_aux import SSLModel, TwoViews, TWO_VIEW_METHODS
    train_tf = build_transform(cfg, train=True)
    ssl_tf = TwoViews(train_tf) if cfg['ssl'] in TWO_VIEW_METHODS else train_tf
    if cfg['ssl'] == 'fixmatch':
        ssl_tf = WeakStrong(train_tf, build_strong_transform(cfg))
    full = datasets.ImageFolder(DATA_ROOT / 'train', transform=ssl_tf if cfg['ssl_source'] == 'train' else train_tf)
    full_eval = datasets.ImageFolder(DATA_ROOT / 'train', transform=build_transform(cfg, train=False))
    val_size = int(round(len(full) * VAL_FRACTION))
    train_split, val_split = random_split(full, [len(full) - val_size, val_size],
                                          generator=torch.Generator().manual_seed(SPLIT_SEED))
    kw = dict(batch_size=cfg['batch_size'], num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    # 'trainval': the validation images join the labelled training set (no selection set left;
    # the logged validation accuracy then measures fit to training data, not generalisation).
    train_idx = train_split.indices + (val_split.indices if cfg['train_on'] in ('trainval', 'all') else [])
    train_set = Subset(full, train_idx)
    if cfg['train_on'] == 'all':
        # Final model: the labelled test images join the training set too (nothing held out).
        test_labelled = datasets.ImageFolder(DATA_ROOT / 'test', transform=full.transform)
        assert test_labelled.classes == full.classes
        train_set = torch.utils.data.ConcatDataset([train_set, test_labelled])
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    val_loader = DataLoader(Subset(full_eval, val_split.indices), shuffle=False, **kw)
    ssl_iter = None
    if cfg['ssl'] != 'none' and cfg['ssl_source'] != 'train':
        # Unlabelled SSL batches from held-out images (labels are never read), with the
        # training augmentation; cycled independently of the labelled training batches.
        if cfg['ssl_source'] == 'val':
            ssl_ds = Subset(datasets.ImageFolder(DATA_ROOT / 'train', transform=ssl_tf), val_split.indices)
        else:
            ssl_ds = UnlabeledImages(DATA_ROOT / cfg['ssl_source'], transform=ssl_tf, classes=full.classes)
        ssl_loader = DataLoader(ssl_ds, shuffle=True, drop_last=True, **kw)

        def cycle(loader):
            while True:
                for xs, ids in loader:
                    yield xs, ids
        diag = torch.tensor(getattr(ssl_ds, 'diag_labels', [-1] * len(ssl_ds)))
        ssl_iter = cycle(ssl_loader)

    set_random_seed(seed)
    model = build_model(cfg, len(full.classes))
    base_model = model                                          # deployed classifier (no SSL heads)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg['ssl'] not in ('none', 'fixmatch'):
        model = SSLModel(model, cfg['ssl'], cfg['img_size'], seed=seed)
    model = model.to(device)
    if cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError(f"unknown optimizer {cfg['optimizer']}")
    criterion = nn.CrossEntropyLoss()
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    @torch.inference_mode()
    def evaluate():
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                logits = model(x)
            logits = logits.float()
            loss_sum += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        return loss_sum / total, correct / total

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_epoch, best_state = 0.0, 0, None
    start = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        ssl_sum = 0.0
        fm_masked, fm_total, fm_correct, fm_labelled = 0, 0, 0, 0
        for x, y in train_loader:
            x2 = None
            if isinstance(x, (list, tuple)):          # two views for contrastive SSL
                x, x2 = x[0].to(device, non_blocking=True), x[1].to(device, non_blocking=True)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                if cfg['ssl'] == 'none':
                    loss = criterion(model(x).float(), y)
                    total = loss
                elif cfg['ssl'] == 'fixmatch':
                    # One forward over [labelled, weak, strong] so batch norm sees all of them.
                    (uw, us), ids = next(ssl_iter)
                    uw, us = uw.to(device, non_blocking=True), us.to(device, non_blocking=True)
                    out = model(torch.cat([x, uw, us])).float()
                    logits, lw, ls = out[:x.size(0)], out[x.size(0):x.size(0) + uw.size(0)], out[x.size(0) + uw.size(0):]
                    conf, pseudo = torch.softmax(lw.detach(), 1).max(1)
                    mask = (conf >= cfg['fixmatch_tau']).float()
                    ssl_loss = (F.cross_entropy(ls, pseudo, reduction='none') * mask).mean()
                    loss = criterion(logits, y)
                    total = loss + cfg['ssl_weight'] * ssl_loss
                    ssl_sum += ssl_loss.item() * x.size(0)
                    fm_masked += int(mask.sum().item()); fm_total += mask.numel()
                    d = diag[ids].to(device)
                    known = (d >= 0) & mask.bool()
                    fm_labelled += int(known.sum().item()); fm_correct += int((pseudo[known] == d[known]).sum().item())
                else:
                    xs = xs2 = None
                    if ssl_iter is not None:
                        xs, _ = next(ssl_iter)
                        if isinstance(xs, (list, tuple)):
                            xs, xs2 = xs[0].to(device, non_blocking=True), xs[1].to(device, non_blocking=True)
                        else:
                            xs = xs.to(device, non_blocking=True)
                    logits, ssl_loss = model.train_step(x, x2, xs, xs2)
                    loss = criterion(logits.float(), y)
                    total = loss + cfg['ssl_weight'] * ssl_loss
                    ssl_sum += ssl_loss.item() * x.size(0)
            total.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_sum += loss.item() * x.size(0)
            seen += y.size(0)
        val_loss, val_acc = evaluate()
        history['train_loss'].append(loss_sum / seen)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history.setdefault('lr', []).append(optimizer.param_groups[0]['lr'])
        if cfg['ssl'] != 'none':
            history.setdefault('ssl_loss', []).append(ssl_sum / seen)
        if cfg['ssl'] == 'fixmatch':
            history.setdefault('fm_mask_rate', []).append(fm_masked / max(fm_total, 1))
            history.setdefault('fm_pseudo_acc', []).append(fm_correct / fm_labelled if fm_labelled else None)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
    @torch.inference_mode()
    def predict(loader):
        model.eval()
        preds = []
        for x, _ in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp']):
                preds.append(model(x.to(device, non_blocking=True)).float().argmax(1).cpu())
        return torch.cat(preds).tolist()

    # Final-epoch model (what the last-10 metric describes), not the best-epoch checkpoint.
    val_labels = [full.targets[i] for i in val_split.indices]
    val_preds = predict(val_loader)
    gray_cfg = {**cfg, 'keep_color': False, 'grayscale': True}
    gray_eval = datasets.ImageFolder(DATA_ROOT / 'train', transform=build_transform(gray_cfg, train=False))
    val_preds_gray = predict(DataLoader(Subset(gray_eval, val_split.indices), shuffle=False, **kw))

    return {'config': cfg, 'seed': seed, 'best_val_acc': best_acc, 'best_epoch': best_epoch,
            'classes': full.classes, 'val_labels': val_labels, 'val_preds': val_preds,
            'val_preds_gray': val_preds_gray,
            'time_s': time.time() - start, 'history': history,
            'params': n_params, 'params_with_ssl_heads': sum(p.numel() for p in model.parameters()),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}, \
        {k: v.detach().cpu() for k, v in base_model.state_dict().items()}   # final-epoch weights


# ----------------------------------------------------------------------------- parallel grid
def full_config(cfg):
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown config keys: {unknown}')
    bad = set(cfg.get('augment', [])) - set(AUGMENTATIONS)
    if bad:
        raise ValueError(f'unknown augmentations: {bad}')
    from ssl_aux import SSL_METHODS
    if cfg.get('train_on', 'train') not in ('train', 'trainval', 'all'):
        raise ValueError(f"unknown train_on {cfg['train_on']}")
    if cfg.get('ssl_source', 'train') not in ('train', 'val', 'test', 'test2'):
        raise ValueError(f"unknown ssl_source {cfg['ssl_source']}")
    if cfg.get('ssl') == 'fixmatch' and cfg.get('ssl_source', 'train') == 'train':
        raise ValueError('fixmatch needs held-out unlabelled images (ssl_source val/test/test2)')
    if cfg.get('ssl', 'none') not in SSL_METHODS:
        raise ValueError(f"unknown ssl method {cfg['ssl']}")
    if cfg.get('schedule', 'constant') not in SCHEDULES:
        raise ValueError(f"unknown schedule {cfg['schedule']}")
    full = {**DEFAULTS, **cfg}
    full['augment'] = sorted(full['augment'])
    return full


def config_id(cfg):
    """Hash of the settings that differ from DEFAULTS, so adding a new option with a
    default value later does not invalidate cached runs."""
    changed = {k: v for k, v in full_config(cfg).items() if DEFAULTS[k] != v}
    return hashlib.sha1(json.dumps(changed, sort_keys=True).encode()).hexdigest()[:10]


def visible_gpus():
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return [g for g in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if g != '']
    out = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                         capture_output=True, text=True)
    return out.stdout.split() or ['']


def run_grid(name, configs, seeds=(0, 1, 2), gpus=None, jobs_per_gpu=2, force=False, save_checkpoint=False):
    """Run every (config, seed) as its own process, `jobs_per_gpu` at a time on each GPU.

    Each finished run is cached in results/runs/<name>/<config-id>_s<seed>.json, so
    re-running a cell only trains what is missing (pass force=True to retrain).
    Returns the run records in the order of `configs` x `seeds`.
    """
    gpus = gpus or visible_gpus()
    out_dir = RESULTS_DIR / 'runs' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(full_config(c), s, out_dir / f'{config_id(c)}_s{s}.json') for c in configs for s in seeds]
    todo = [j for j in jobs if force or not j[2].exists() or (save_checkpoint and not j[2].with_suffix('.pt').exists())]
    print(f'{name}: {len(jobs)} runs ({len(jobs) - len(todo)} cached), '
          f'GPUs {gpus} x {jobs_per_gpu} slots', flush=True)

    slots = queue.Queue()
    for _ in range(jobs_per_gpu):
        for g in gpus:
            slots.put(g)

    def launch(job):
        cfg, seed, path = job
        gpu = slots.get()
        try:
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--config', json.dumps(cfg), '--seed', str(seed), '--out', str(path)]
                                  + (['--save-checkpoint'] if save_checkpoint else []),
                                  env=env, capture_output=True, text=True, cwd=ROOT)
            if proc.returncode != 0:
                raise RuntimeError(f'run failed (gpu {gpu}, seed {seed}, {cfg}):\n{proc.stderr[-2000:]}')
            rec = json.loads(path.read_text())
            changed = {k: v for k, v in cfg.items() if DEFAULTS[k] != v}
            print(f"  done gpu{gpu} seed {seed} val {rec['best_val_acc']:.4f}  {changed or 'baseline'}", flush=True)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus) * jobs_per_gpu) as pool:
        list(pool.map(launch, todo))
    return [json.loads(p.read_text()) for _, _, p in jobs]


def summarize(name, runs, extra=None):
    """Mean +- std of best validation accuracy over seeds (in %)."""
    accs = np.array([r['best_val_acc'] for r in runs]) * 100
    row = {'experiment': name, 'val_acc_mean': float(accs.mean()), 'val_acc_std': float(accs.std()),
           'per_seed': accs.round(2).tolist(), 'best_epochs': [r['best_epoch'] for r in runs],
           'params': runs[0]['params'], 'time_s_per_run': float(np.mean([r['time_s'] for r in runs])),
           **(extra or {})}
    print(f"{name:32s} val acc {row['val_acc_mean']:5.2f} ± {row['val_acc_std']:4.2f}  "
          f"seeds {row['per_seed']}  best epochs {row['best_epochs']}  "
          f"params {row['params']:,}  {row['time_s_per_run']:.1f}s/run")
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON dict overriding DEFAULTS')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--save-checkpoint', action='store_true')
    args = parser.parse_args()
    cfg = full_config(json.loads(args.config))
    record, state = train_one(cfg, args.seed)   # state = final-epoch weights of the classifier
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoint:
        import torch
        torch.save({'config': cfg, 'seed': args.seed, 'state_dict': state, 'epoch': cfg['epochs'],
                    'classes': record['classes']}, out.with_suffix('.pt'))
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out)       # atomic: a cached result is never half-written
