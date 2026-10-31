"""Self-supervised pretraining on the training images only (Assignment 2).

Methods
  simclr : contrastive. Two random views of the same image are pulled together, views of other
           images in the batch are pushed apart (NT-Xent loss, temperature 0.2).
  moco   : MoCo v2. Contrastive, but the keys come from a momentum (EMA) copy of the encoder and the
           negatives from a queue of past keys, so the number of negatives does not depend on batch size.
  dino   : self-distillation without labels. A student network matches the output distribution of an
           EMA teacher across views (2 global + 6 local crops); centring + sharpening of the teacher
           output prevents collapse.
  mae    : masked autoencoder. A ViT encoder sees 25% of the image patches; a light decoder reconstructs
           the 75% masked patches (per-patch normalised pixels, loss on masked patches only).

Data: only the 1,920 images of the training split (seed-0 split shared with cvexp.py); labels are never
read. No external data or weights: every network is randomly initialised.

    python ssl_pretrain.py --method simclr --epochs 400 --out results/ssl/simclr_resnet18.pt
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from cvexp import DATA_ROOT, SPLIT_SEED, VAL_FRACTION

MEAN, STD = [0.5] * 3, [0.5] * 3


# ----------------------------------------------------------------------------- augmentations
def ssl_view(size, scale=(0.2, 1.0), blur_p=0.5, solarize_p=0.0):
    """SimCLR-style view for grayscale images (colour jitter reduced to brightness/contrast)."""
    k = int(0.1 * size) // 2 * 2 + 1
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(k, sigma=(0.1, 2.0))], p=blur_p),
        transforms.RandomSolarize(threshold=128, p=solarize_p),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class MultiView:
    def __init__(self, tfs):
        self.tfs = tfs

    def __call__(self, img):
        return [t(img) for t in self.tfs]


def view_transform(method, size):
    if method in ('simclr', 'moco'):
        return MultiView([ssl_view(size), ssl_view(size)])
    if method == 'dino':
        small = size // 2
        return MultiView([ssl_view(size, (0.25, 1.0), blur_p=1.0),
                          ssl_view(size, (0.25, 1.0), blur_p=0.1, solarize_p=0.2)] +
                         [ssl_view(small, (0.05, 0.25), blur_p=0.5) for _ in range(6)])
    if method == 'mae':
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    raise ValueError(method)


# ----------------------------------------------------------------------------- backbones
def build_backbone(arch, img_size):
    """Randomly initialised encoder returning a feature vector; (module, feature dim)."""
    if arch == 'resnet18':
        import torchvision
        net = torchvision.models.resnet18(weights=None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim
    if arch == 'vit_tiny':
        import timm
        net = timm.create_model('vit_tiny_patch16_224', pretrained=False, img_size=img_size, num_classes=0)
        return net, net.num_features
    raise ValueError(arch)


def mlp(in_dim, hidden, out_dim, last_bn=False):
    layers = [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)]
    if last_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- methods
def nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.size(0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float('-inf'))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


class SimCLR(nn.Module):
    def __init__(self, backbone, dim, tau=0.2):
        super().__init__()
        self.backbone, self.head, self.tau = backbone, mlp(dim, 2048, 128), tau

    def forward(self, views, step_frac):
        x1, x2 = views
        z = self.head(self.backbone(torch.cat([x1, x2])))
        return nt_xent(*z.chunk(2), self.tau)


class MoCo(nn.Module):
    """MoCo v2: query encoder + EMA key encoder, queue of negatives, InfoNCE."""
    def __init__(self, backbone, dim, tau=0.2, queue=1024, momentum=0.99):
        super().__init__()
        self.backbone, self.head = backbone, mlp(dim, 2048, 128)
        self.k_backbone, self.k_head = copy.deepcopy(self.backbone), copy.deepcopy(self.head)
        for p in list(self.k_backbone.parameters()) + list(self.k_head.parameters()):
            p.requires_grad = False
        self.tau, self.m = tau, momentum
        self.register_buffer('queue', F.normalize(torch.randn(queue, 128), dim=1))
        self.register_buffer('ptr', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self, step_frac):
        for q, k in zip(list(self.backbone.parameters()) + list(self.head.parameters()),
                        list(self.k_backbone.parameters()) + list(self.k_head.parameters())):
            k.mul_(self.m).add_(q.detach(), alpha=1 - self.m)

    def forward(self, views, step_frac):
        x1, x2 = views
        q = F.normalize(self.head(self.backbone(x1)).float(), dim=1)
        with torch.no_grad():
            k = F.normalize(self.k_head(self.k_backbone(x2)).float(), dim=1)
        logits = torch.cat([(q * k).sum(1, keepdim=True), q @ self.queue.t()], 1) / self.tau
        loss = F.cross_entropy(logits, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
        with torch.no_grad():                                  # enqueue keys, dequeue oldest
            n, p = k.size(0), int(self.ptr)
            idx = (torch.arange(n, device=k.device) + p) % self.queue.size(0)
            self.queue[idx] = k
            self.ptr.fill_((p + n) % self.queue.size(0))
        return loss


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class DINO(nn.Module):
    """Student/teacher self-distillation with multi-crop, centring and sharpening (Caron et al. 2021)."""
    def __init__(self, backbone, dim, out_dim=4096, t_student=0.1, t_teacher=0.04, center_m=0.9,
                 m_start=0.996):
        super().__init__()
        self.backbone, self.head = backbone, DINOHead(dim, out_dim)
        self.t_backbone, self.t_head = copy.deepcopy(self.backbone), copy.deepcopy(self.head)
        for p in list(self.t_backbone.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.ts, self.tt, self.cm, self.m_start = t_student, t_teacher, center_m, m_start
        self.register_buffer('center', torch.zeros(1, out_dim))

    @torch.no_grad()
    def momentum_update(self, step_frac):
        m = 1 - (1 - self.m_start) * (math.cos(math.pi * step_frac) + 1) / 2     # 0.996 -> 1
        for s, t in zip(list(self.backbone.parameters()) + list(self.head.parameters()),
                        list(self.t_backbone.parameters()) + list(self.t_head.parameters())):
            t.mul_(m).add_(s.detach(), alpha=1 - m)

    def forward(self, views, step_frac):
        glob, loc = views[:2], views[2:]
        b = glob[0].size(0)
        with torch.no_grad():
            t_out = self.t_head(self.t_backbone(torch.cat(glob))).float()
            t_prob = F.softmax((t_out - self.center) / self.tt, dim=-1).chunk(2)
        s_out = [self.head(self.backbone(torch.cat(glob)))] + [self.head(self.backbone(torch.cat(loc)))]
        s_logp = torch.cat([o.float() for o in s_out]).div(self.ts).log_softmax(-1).split(b)
        loss, n = 0.0, 0
        for ti, t in enumerate(t_prob):
            for si, s in enumerate(s_logp):
                if si == ti:                                    # same view: skip
                    continue
                loss = loss + (-t * s).sum(-1).mean()
                n += 1
        with torch.no_grad():
            self.center.mul_(self.cm).add_(t_out.mean(0, keepdim=True), alpha=1 - self.cm)
        return loss / n


class MAE(nn.Module):
    """Masked autoencoder on a timm ViT; the encoder keeps timm's module names so its weights load
    directly into the fine-tuning ViT."""
    def __init__(self, vit, mask_ratio=0.75, dec_dim=128, dec_depth=2, dec_heads=4):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.vit, self.mask_ratio = vit, mask_ratio
        self.patch = vit.patch_embed.patch_size[0]
        n = vit.patch_embed.num_patches
        self.dec_embed = nn.Linear(vit.embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, n + 1, dec_dim))
        self.dec_blocks = nn.Sequential(*[Block(dec_dim, dec_heads, 4.0) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.patch ** 2 * 3)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.dec_pos, std=0.02)

    def patchify(self, x):
        p = self.patch
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, (h // p) * (w // p), p * p * c)

    def forward(self, x, step_frac):
        v = self.vit
        tokens = v.patch_embed(x) + v.pos_embed[:, 1:]
        b, n, d = tokens.shape
        keep = int(n * (1 - self.mask_ratio))
        order = torch.rand(b, n, device=x.device).argsort(1)
        restore = order.argsort(1)
        vis = torch.gather(tokens, 1, order[:, :keep, None].expand(-1, -1, d))
        cls = (v.cls_token + v.pos_embed[:, :1]).expand(b, -1, -1)
        h = v.norm(v.blocks(torch.cat([cls, vis], 1)))
        h = self.dec_embed(h)
        mask_tok = self.mask_token.expand(b, n - keep, -1)
        full = torch.gather(torch.cat([h[:, 1:], mask_tok], 1), 1, restore[..., None].expand(-1, -1, h.size(-1)))
        full = torch.cat([h[:, :1], full], 1) + self.dec_pos
        pred = self.dec_pred(self.dec_norm(self.dec_blocks(full)))[:, 1:].float()
        target = self.patchify(x).float()
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, restore)                    # 1 = masked patch
        return (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum()


DEFAULT_HP = {   # per-method optimiser settings (standard choices for small-scale SSL; not tuned on test)
    'simclr': dict(arch='resnet18', opt='sgd', lr=0.3, wd=5e-4, batch=256, warmup=10),
    'moco':   dict(arch='resnet18', opt='sgd', lr=0.3, wd=5e-4, batch=256, warmup=10),
    'dino':   dict(arch='resnet18', opt='adamw', lr=5e-4, wd=0.04, batch=128, warmup=10),
    'mae':    dict(arch='vit_tiny', opt='adamw', lr=1.5e-4, wd=0.05, batch=256, warmup=40),
}


def train_split_indices(n_total):
    val = int(round(n_total * VAL_FRACTION))
    tr, _ = random_split(range(n_total), [n_total - val, val], generator=torch.Generator().manual_seed(SPLIT_SEED))
    return tr.indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=list(DEFAULT_HP))
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--images', default='train', choices=['train', 'all'],
                    help="'train': the 1,920 training-split images; 'all': all 2,400 training images (final model)")
    args = ap.parse_args()
    hp = DEFAULT_HP[args.method]
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

    full = datasets.ImageFolder(DATA_ROOT / 'train', transform=view_transform(args.method, args.img_size))
    ds = Subset(full, train_split_indices(len(full))) if args.images == 'train' else full   # labels unused
    loader = DataLoader(ds, batch_size=hp['batch'], shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)

    backbone, dim = build_backbone(hp['arch'], args.img_size)
    model = {'simclr': lambda: SimCLR(backbone, dim), 'moco': lambda: MoCo(backbone, dim),
             'dino': lambda: DINO(backbone, dim), 'mae': lambda: MAE(backbone)}[args.method]().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = hp['lr'] * hp['batch'] / 256
    if hp['opt'] == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=hp['wd'])
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=hp['wd'], betas=(0.9, 0.95))
    total = args.epochs * len(loader)
    warm = hp['warmup'] * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else
                                              0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))
    history, step, start = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for views, _ in loader:
            views = [v.to(device, non_blocking=True) for v in views] if isinstance(views, list) else views.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(views, step / total)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.method == 'dino':
                torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            sched.step()
            if hasattr(model, 'momentum_update'):
                model.momentum_update(step / total)
            tot += loss.item(); nb += 1; step += 1
        history.append(tot / nb)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f'epoch {epoch:4d} loss {tot / nb:.4f}  {(time.time() - start) / 60:.1f} min', flush=True)
    enc = model.vit if args.method == 'mae' else model.backbone      # student / query encoder
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'method': args.method, 'arch': hp['arch'], 'img_size': args.img_size, 'epochs': args.epochs,
                'hp': hp, 'seed': args.seed, 'images': args.images, 'state_dict': {k: v.cpu() for k, v in enc.state_dict().items()},
                'loss_history': history, 'time_s': time.time() - start}, out)
    json.dump({'method': args.method, 'loss_history': history, 'time_s': time.time() - start},
              open(out.with_suffix('.json'), 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
