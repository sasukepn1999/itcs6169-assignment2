"""Self-supervised auxiliary tasks for multi-task training (Experiment 6).

Total loss = cross-entropy(classification) + ssl_weight * SSL loss. Classification always uses
the labelled training images; the SSL loss uses the training images (ssl_source='train',
Experiment 6) or the unlabelled validation images (ssl_source='val', Experiment 7). The SSL
heads are used only during training. `SSLModel.forward` is exactly the original network, so evaluation
and prediction are unchanged.

Methods (all CNN-compatible):
  rotation : predict which of 0/90/180/270 degrees a copy of the image was rotated by.
  simclr   : NT-Xent contrastive loss between two augmented views (temperature 0.2).
  simsiam  : SimSiam / BYOL-style negative-cosine loss between two views with a predictor
             and stop-gradient (no negatives, no momentum encoder).
  mim      : masked image modelling (MAE-style for CNNs): mask 60% of 20x20-px patches
             (10x10 grid at 200 px), reconstruct the masked pixels with a light upsampling
             conv decoder on the last feature map; MSE on masked pixels only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SSL_METHODS = ['none', 'rotation', 'simclr', 'simsiam', 'mim', 'fixmatch']   # fixmatch: see cvexp.py
TWO_VIEW_METHODS = {'simclr', 'simsiam'}


class TwoViews:
    """Apply the (random) training transform twice to get two views of one image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)

    def __repr__(self):
        return f'TwoViews({self.transform})'


def split_backbone(model):
    """(feature-map extractor, pooled feature dim, classifier) for supported torchvision CNNs."""
    name = type(model).__name__
    if name == 'EfficientNet':
        return model.features, model.classifier[-1].in_features, model.classifier
    if name == 'ResNet':
        feats = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                              model.layer1, model.layer2, model.layer3, model.layer4)
        return feats, model.fc.in_features, model.fc
    raise ValueError(f'SSL auxiliary tasks not implemented for {name}')


class SSLModel(nn.Module):
    def __init__(self, base, method, img_size, mask_ratio=0.6, grid=10, temperature=0.2, seed=0):
        super().__init__()
        self.method, self.img_size = method, img_size
        self.mask_ratio, self.grid, self.temperature = mask_ratio, grid, temperature
        self.features, d, self.classifier = split_backbone(base)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if method == 'rotation':
            self.head = nn.Linear(d, 4)
        elif method == 'simclr':
            self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
        elif method == 'simsiam':
            self.head = nn.Sequential(nn.Linear(d, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                      nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512, affine=False))
            self.predictor = nn.Sequential(nn.Linear(512, 128, bias=False), nn.BatchNorm1d(128),
                                           nn.ReLU(inplace=True), nn.Linear(128, 512))
        elif method == 'mim':
            chans = [256, 128, 64, 32, 16]
            layers = [nn.Conv2d(d, chans[0], 1)]
            for cin, cout in zip(chans[:-1], chans[1:]):
                layers += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(cin, cout, 3, padding=1),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(chans[-1], 1, 3, padding=1)]
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError(f'unknown SSL method {method}')
        # Dedicated RNG for rotations / masks, so the SSL sampling does not shift the global
        # RNG streams used for initialisation and dropout.
        self._gen_seed = seed + 12345
        self._gen = None

    def _rng(self, device):
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device).manual_seed(self._gen_seed)
        return self._gen

    def embed(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x):
        """Plain classification forward pass (identical to the original torchvision model)."""
        return self.classifier(self.embed(x))

    def train_step(self, x, x2=None, xs=None, xs2=None):
        """Return (classification logits, SSL loss) for one step.

        Classification always uses the labelled batch `x`. The SSL loss uses `x`/`x2` by default
        (SSL on the training images); if `xs` is given, it uses the unlabelled batch `xs`/`xs2`
        instead (e.g. SSL on the validation images, `ssl_source='val'`).
        """
        z = self.embed(x)
        logits = self.classifier(z)
        if xs is not None:
            x, x2 = xs, xs2
            z = self.embed(x) if self.method in ('simclr', 'simsiam') else None
        g = self._rng(x.device)
        if self.method == 'rotation':
            k = torch.randint(0, 4, (x.size(0),), device=x.device, generator=g)
            xr = x.clone()
            for r in range(1, 4):
                sel = k == r
                if sel.any():
                    xr[sel] = torch.rot90(x[sel], r, dims=(2, 3))
            ssl = F.cross_entropy(self.head(self.embed(xr)).float(), k)
        elif self.method == 'simclr':
            h = F.normalize(torch.cat([self.head(z), self.head(self.embed(x2))]).float(), dim=1)
            n = x.size(0)
            sim = h @ h.t() / self.temperature
            sim.fill_diagonal_(float('-inf'))
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(x.device)
            ssl = F.cross_entropy(sim, target)
        elif self.method == 'simsiam':
            z1, z2 = self.head(z), self.head(self.embed(x2))
            p1, p2 = self.predictor(z1), self.predictor(z2)
            ssl = -0.5 * (F.cosine_similarity(p1.float(), z2.detach().float()).mean() +
                          F.cosine_similarity(p2.float(), z1.detach().float()).mean())
        elif self.method == 'mim':
            b, _, hgt, wid = x.shape
            keep = torch.rand(b, 1, self.grid, self.grid, device=x.device, generator=g) >= self.mask_ratio
            mask = 1.0 - F.interpolate(keep.float(), size=(hgt, wid), mode='nearest')   # 1 = masked
            recon = F.interpolate(self.head(self.features(x * (1 - mask))), size=(hgt, wid),
                                  mode='bilinear', align_corners=False).float()
            target = x.mean(1, keepdim=True).float()          # gray image (channels are identical)
            ssl = ((recon - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return logits, ssl
