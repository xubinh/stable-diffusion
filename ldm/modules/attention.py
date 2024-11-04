import math
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from ldm.modules.diffusionmodules.util import checkpoint


# 没有使用过:
def _make_unique(array: Sequence[Any]) -> List[Any]:
    # return {element: True for element in array}.keys()
    return list(set(array))


T = TypeVar("T")


def get_default_if_not_exists(
    value: Optional[T],
    default_value_or_callable: Union[Callable[[], T], T],
) -> T:
    if value is not None:
        return value

    if callable(default_value_or_callable):
        return default_value_or_callable()

    return default_value_or_callable


def get_max_negative_value(t) -> float:
    return -torch.finfo(t.dtype).max


# 没有使用过:
def _initialize_tensor(tensor: torch.Tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)

    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = int(dim * mult)
        dim_out = get_default_if_not_exists(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """

    for p in module.parameters():
        p.detach().zero_()

    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


# 没有使用过:
class _SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        # w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        number_of_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = number_of_heads * head_dim
        context_dim = get_default_if_not_exists(context_dim, query_dim)

        self.scale = head_dim**-0.5
        self.number_of_heads = number_of_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,  # batch_size x pixel_number x latent_dim
        context_tensor: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # 如果不给出上下文 embedding, 那么 cross-attention 就退化为 self-attention:
        context_tensor = get_default_if_not_exists(context_tensor, x)

        h = self.number_of_heads

        q = self.to_q(x)  # b n l -> b n (h d)
        k = self.to_k(context_tensor)  # b n t -> b n (h d)
        v = self.to_v(context_tensor)  # b n t -> b n (h d)

        q, k, v = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        attention = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            max_negative_value = -torch.finfo(attention.dtype).max
            attention.masked_fill_(~mask, max_negative_value)  # type: ignore

        # attention, what we cannot get enough of
        attention = attention.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attention, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)  # b n (h d) -> b n l

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        number_of_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff=True,
        checkpoint: bool = True,
    ):
        super().__init__()

        self.attn1 = CrossAttention(
            query_dim=dim,
            number_of_heads=number_of_heads,
            head_dim=head_dim,
            dropout=dropout,
        )  # is a self-attention

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            number_of_heads=number_of_heads,
            head_dim=head_dim,
            dropout=dropout,
        )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels: int,
        number_of_heads: int,
        head_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = number_of_heads * head_dim
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    number_of_heads,
                    head_dim,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention

        b, c, h, w = x.shape

        x_in = x

        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.transformer_blocks:
            x = block(x, context=context)

        # 此时 x 富含其自身以及文本嵌入的语义

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = self.proj_out(x)  # 一开始为零

        return x + x_in  # residual 连接
