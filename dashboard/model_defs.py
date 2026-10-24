"""Model definitions copied verbatim from the training notebook (CELL 6 and CELL 13).
Keep these identical to the notebook, otherwise the saved checkpoints will not load."""
import torch, torch.nn as nn, torch.nn.functional as F
import timm
try:
    from timm.layers import LayerNorm2d
except Exception:
    from timm.models.layers import LayerNorm2d


class CBAM2d(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True), nn.Linear(c // r, c, bias=False))
        self.sp = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))[:, :, None, None]
        x = x * ca
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.sp(s))


class GCBlock2d(nn.Module):
    def __init__(self, c, ratio=4):
        super().__init__()
        inner = c // ratio
        self.ctx = nn.Conv2d(c, 1, 1)
        self.transform = nn.Sequential(nn.Conv2d(c, inner, 1), nn.GroupNorm(1, inner), nn.ReLU(inplace=True), nn.Conv2d(inner, c, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        attn = torch.softmax(self.ctx(x).flatten(2), dim=-1)
        ctx = torch.bmm(x.flatten(2), attn.transpose(1, 2)).unsqueeze(-1)
        return x + self.transform(ctx)


class Stage1Net(nn.Module):
    def __init__(self, pretrained=False, drop_path=0.2, emb_dim=512, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        C = self.backbone.num_features
        self.norm = LayerNorm2d(C); self.cbam = CBAM2d(C); self.gc = GCBlock2d(C)
        self.embed_fc = nn.Sequential(nn.Linear(C, emb_dim), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.fc1 = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(inplace=True), nn.Dropout(drop))
        self.fc2 = nn.Linear(128, 2)

    def embed(self, x):
        f = self.backbone.forward_features(x)
        assert f.ndim == 4, f'unexpected backbone output {tuple(f.shape)}'
        f = self.gc(self.cbam(self.norm(f)))
        return self.embed_fc(f.mean((2, 3)))

    def forward(self, x):
        return self.fc2(self.fc1(self.embed(x)))


class DeviationGate(nn.Module):
    def __init__(self, dim=512, hidden=256, n_buckets=3, mode='conditioned', momentum=0.99):
        super().__init__()
        self.mode, self.m, self.nb = mode, momentum, n_buckets
        self.register_buffer('E_bucket', torch.zeros(n_buckets, dim)); self.register_buffer('E_global', torch.zeros(dim))
        self.register_buffer('ok_bucket', torch.zeros(n_buckets, dtype=torch.bool)); self.register_buffer('ok_global', torch.zeros((), dtype=torch.bool))
        self.comparator = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, hidden), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(hidden, dim))
        nn.init.constant_(self.comparator[-1].bias, 2.0)

    def reference(self, bucket_probs):
        if self.mode == 'global':
            return self.E_global.unsqueeze(0).expand(bucket_probs.shape[0], -1)
        return bucket_probs @ self.E_bucket

    def forward(self, A, bucket_probs):
        E = self.reference(bucket_probs)
        D = torch.sigmoid(self.comparator(torch.cat([A, E, A - E, A * E], 1)))
        return A * D, D


class VerificationNet(nn.Module):
    def __init__(self, mode='conditioned', dim=512, momentum=0.99):
        super().__init__()
        self.mode = mode
        self.bucket_head = nn.Linear(dim, 3)
        self.gate = None if mode == 'nogate' else DeviationGate(dim, 256, 3, mode, momentum)
        self.cdr_head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, A, bucket=None):
        blog = self.bucket_head(A); soft = torch.softmax(blog, 1)
        probs = soft  # deployable path: the disc-size bucket is predicted (no mask at inference)
        if self.gate is None:
            A2, D = A, None
        else:
            A2, D = self.gate(A, probs)
        return self.cdr_head(A2).squeeze(1), blog, D
