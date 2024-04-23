from enum import Enum
import functools
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat



# constants

class TokenTypes(Enum):
    S1 = 0
    S2 = 1
    DEM = 2
    DNW = 3
    FUSION = 4


# functions

def exists(val):
    return val is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor


def pair(t):
    return (t, t) if not isinstance(t, tuple) else t


def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)


def divisible_by(numer, denom):
    return (numer % denom) == 0


# decorators

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


# attention
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            context=None,
            attn_mask=None
    ):
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Block(nn.Module):

    def __init__(self, dim=768, dim_head=64, heads=8, ff_mult=4, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, attn_mask):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_Fusion(nn.Module):

    def __init__(self, dim=768, dim_head=64, heads=8, ff_mult=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.mlp = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, attn_mask):
        B, _, _, _ = x.shape
        x = rearrange(x, 'b n m d -> (b n) m d')
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = rearrange(x[:, -1, :].squeeze(1), '(b n) d -> b n d', b=B)
        x = x + self.mlp(self.norm2(x))
        return x

class Attention_LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(Attention_LSTM, self).__init__()

        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, H, mask=None):
        M = torch.tanh(H)
        M = self.attention(M).squeeze(2)
        if mask is not None:
            M = M.masked_fill(mask == 0, -1e+4)
        alpha = F.softmax(M, dim=1).unsqueeze(1)
        return alpha


class AttentionBiLSTM(nn.Module):
    def __init__(self, embedding_dim, num_layers=1, dropout=0.0, emb_layer_dropout=0.0):
        super(AttentionBiLSTM, self).__init__()
        # embedding layer dropout
        self.embedding_dim = embedding_dim
        #self.emb_layer_dropout = nn.Dropout(emb_layer_dropout)
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            embedding_dim,
                            num_layers,
                            dropout=(0 if num_layers == 1 else dropout),
                            bidirectional=True,
                            batch_first=True)
        # penultimate layer
        self.attention = Attention_LSTM(embedding_dim)

    def forward(self, embedded, mask=None):
        #embedded = self.emb_layer_dropout(embedded)
        y, _ = self.lstm(embedded)
        y = y[:, :, :self.embedding_dim] + y[:, :, self.embedding_dim:]
        alpha = self.attention(y, mask)
        r = alpha.bmm(y).squeeze(1)
        #h = torch.tanh(r)
        return r
