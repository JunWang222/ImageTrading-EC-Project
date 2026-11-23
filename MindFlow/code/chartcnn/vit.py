"""Small-data vision transformer for chart images, in the style of Tokens-to-Token ViT
(Yuan et al., ICCV 2021): tokens are built by progressive *soft splits* (overlapping patches whose
neighbouring pixels are merged step by step), then a deep-narrow transformer backbone.

    image [1, H, 234]  (H = 234, or 468 with the SPY panel)
    soft split 1  unfold 7x7, stride 3x3   -> (H/3) x 78 tokens; 49 -> 32 per-token MLP
                  (stride 3 = one bar per token column, like the CNN's first layer)
    soft split 2  unfold 3x3, stride 2     -> (H/6) x 39 tokens; 288 -> 64, one token-transformer layer
    soft split 3  unfold 3x3, stride 3     -> (H/18) x 13 tokens; 576 -> dim
    + class token + learned position embedding
    backbone      `depth` pre-norm transformer blocks (dim, heads, MLP ratio), stochastic depth
    head          LayerNorm(class token) -> Dropout -> Linear(2), zero-initialised (P = 0.5 at start)

T2T-ViT uses attention in the first soft-split stage too; at 78 x 78 = 6,084 tokens that is too slow
for a Colab budget, so stage 1 uses a per-token MLP and only stage 2 (1,521 tokens) uses attention.
The defaults give ~0.86M parameters, the same order as the CNNs (0.72-0.94M).
"""
import torch
from torch import nn


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(LN(x)), then x + MLP(LN(x)); optional stochastic depth."""

    def __init__(self, dim, heads, mlp_ratio, drop_path=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.heads = heads
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = drop_path

    def _drop(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v)        # flash / memory-efficient kernel
        x = x + self._drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return x + self._drop(self.mlp(self.n2(x)))


class T2TTokenizer(nn.Module):
    """Image -> (batch, tokens, dim) by three soft splits (see module docstring)."""

    def __init__(self, dim, input_hw):
        super().__init__()
        self.ss1 = nn.Unfold((7, 7), stride=(3, 3), padding=(3, 2))
        self.mlp1 = nn.Sequential(nn.Linear(49, 32), nn.GELU())
        self.ss2 = nn.Unfold((3, 3), stride=(2, 2), padding=(1, 1))
        self.proj2 = nn.Linear(32 * 9, 64)
        self.t2t = Block(64, 1, 1.0)
        self.ss3 = nn.Unfold((3, 3), stride=(3, 3))
        self.proj3 = nn.Linear(64 * 9, dim)
        with torch.no_grad():
            self.grids = self._grids(torch.zeros(1, 1, *input_hw))

    @staticmethod
    def _out(size, k, s, p):
        return (size + 2 * p - k) // s + 1

    def _grids(self, x):
        h, w = x.shape[-2:]
        h1, w1 = self._out(h, 7, 3, 3), self._out(w, 7, 3, 2)
        h2, w2 = self._out(h1, 3, 2, 1), self._out(w1, 3, 2, 1)
        h3, w3 = self._out(h2, 3, 3, 0), self._out(w2, 3, 3, 0)
        return (h1, w1), (h2, w2), (h3, w3)

    def forward(self, x):
        (h1, w1), (h2, w2), _ = self.grids
        b = x.shape[0]
        t = self.mlp1(self.ss1(x).transpose(1, 2))                        # (b, h1*w1, 32)
        t = t.transpose(1, 2).reshape(b, 32, h1, w1)
        t = self.t2t(self.proj2(self.ss2(t).transpose(1, 2)))              # (b, h2*w2, 64)
        t = t.transpose(1, 2).reshape(b, 64, h2, w2)
        return self.proj3(self.ss3(t).transpose(1, 2))                     # (b, h3*w3, dim)


class ClassAndPosition(nn.Module):
    def __init__(self, dim, tokens):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, t):
        return torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos


class ClassToken(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, t):
        return self.norm(t[:, 0])


class ChartViT(nn.Module):
    """T2T-style ViT on chart images. `features` runs image -> class-token embedding and `head`
    maps it to two logits, the same interface as ChartCNN (parameter_table walks both)."""

    def __init__(self, dim=96, depth=8, heads=4, mlp_ratio=3.0, drop_path=0.1, dropout=0.1,
                 input_hw=(234, 234), arch="vit"):
        super().__init__()
        self.arch, self.input_shape = arch, (1, *input_hw)
        tok = T2TTokenizer(dim, input_hw)
        (h3, w3) = tok.grids[2]
        rates = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.features = nn.Sequential(tok, ClassAndPosition(dim, h3 * w3),
                                      *[Block(dim, heads, mlp_ratio, r) for r in rates], ClassToken(dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)       # start at P = 0.5, like the CNNs

    def _on_my_device(self, x):
        return x.to(self.head[-1].weight.device, non_blocking=True)

    def embed(self, x):
        return self.features(self._on_my_device(x))

    def forward(self, x):
        return self.head(self.features(self._on_my_device(x)))
