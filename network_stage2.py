import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.parameter import Parameter
from collections import namedtuple
import torch
from torch.nn import functional
ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings'])

def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [int(((image_size[0] - D.shape[-2]) / stride) + 1),
                     int(((image_size[1] - D.shape[-1]) / stride) + 1)]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = functional.conv_transpose2d(x, D, stride=stride)
        x = functional.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))


def calc_pad_sizes(I: torch.Tensor, kernel_size: int, stride: int):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - (
                (I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - (
                (I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride
    return left_pad, right_pad, top_pad, bot_pad


class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1, size, 1, 1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out
class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = torch.randn(params.num_filters, 31, params.kernel_size, params.kernel_size)
            l = conv_power_method(A, [512, 512], num_iters=200, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)
        self.apply_A = torch.nn.ConvTranspose2d(params.num_filters, 31, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_B = torch.nn.Conv2d(31, params.num_filters, kernel_size=params.kernel_size, stride=params.stride,
                                       bias=False)
        self.apply_C = torch.nn.ConvTranspose2d(params.num_filters, 31, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False)
        self.apply_A.weight.data = A
        self.apply_B.weight.data = B
        self.apply_C.weight.data = C
        self.soft_threshold = SoftThreshold(params.num_filters, threshold)
        self.params = params

    def _split_image(self, I):
        if self.params.stride == 1:
            return I, torch.ones_like(I)
        left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(I, self.params.kernel_size, self.params.stride)
        I_batched_padded = torch.zeros(I.shape[0], self.params.stride ** 2, I.shape[1], top_pad + I.shape[2] + bot_pad,
                                       left_pad + I.shape[3] + right_pad).type_as(I)
        valids_batched = torch.zeros_like(I_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
                [(i, j) for i in range(self.params.stride) for j in range(self.params.stride)]):
            I_padded = functional.pad(I, pad=(
                left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
            valids = functional.pad(torch.ones_like(I), pad=(
                left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
            I_batched_padded[:, num, :, :, :] = I_padded
            valids_batched[:, num, :, :, :] = valids
        I_batched_padded = I_batched_padded.reshape(-1, *I_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return I_batched_padded, valids_batched

    def forward(self, I):
        I_batched_padded, valids_batched = self._split_image(I)
        conv_input = self.apply_B(I_batched_padded)
        gamma_k = self.soft_threshold(conv_input)
        for k in range(self.params.unfoldings - 1):
            x_k = self.apply_A(gamma_k)
            r_k = self.apply_B(x_k - I_batched_padded)
            gamma_k = self.soft_threshold(gamma_k - r_k)
        output_all = self.apply_C(gamma_k)
        # output_cropped = torch.masked_select(output_all, valids_batched.byte()).reshape(I.shape[0],
        #                                                                                 self.params.stride ** 2,
        #                                                                                 *I.shape[1:])
        # # if self.return_all:
        # #     return output_cropped
        # output = output_cropped.mean(dim=1, keepdim=False)
        return output_all

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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CrossWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        kv = self.qkv1(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        qq = self.qkv2(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qq[0]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class CrossSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        y = self.norm2(y)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops




class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.kernel1 = nn.Sequential(nn.Linear(256, dim * 2, bias=False),)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, guided_fea):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        k_v = self.kernel1(guided_fea.squeeze(-1).squeeze(-1)).view(-1, C * 2, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x + self.drop_path(self.mlp(k_v2.permute(0, 2, 1) + k_v1.permute(0, 2, 1) * self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class DualCrossTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        self.cross1 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)

        self.cross2 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, window_size=window_size,
                                                shift_size=shift_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer)

        self.swin = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, window_size=window_size,
                                                shift_size=shift_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer)

    def forward(self, x, y, x_size, guided_fea):
        dual1 = self.cross1(x, y, x_size)
        dual2 = self.cross2(y, x, x_size)
        dual = dual1 + dual2
        out = self.swin(dual, x_size,guided_fea)
        return out


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DualCrossTransformerBlock(dim=dim, input_resolution=input_resolution,depth=depth,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y, x_size, guided_fea):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, y, x_size, guided_fea)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, y, x_size, guided_fea):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, y, x_size, guided_fea), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class dualTransformer(nn.Module):
    def __init__(self, n_feats, img_size=64, patch_size=4, depths=[6,6,6], num_heads=[6,6,6],
                 window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(dualTransformer, self).__init__()
        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.w_gen = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(n_feats, 31, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(31, 31, 1, 1, 0),
        )

        self.u_gen = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(n_feats, 31, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(31, 31, 1, 1, 0),
        )

        self.mlp_ratio = mlp_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(1):
            layer = RSTB(dim=n_feats,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)

        self.norm = norm_layer(n_feats)

    def forward_features(self, x, y, guided_fea):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)

        for i, layer in enumerate(self.layers):
            x = layer(x, y, x_size, guided_fea)


        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, y, guided_fea):
        res = x
        x = self.forward_features(x, y, guided_fea)
        w = self.w_gen(x)
        x = self.u_gen(self.conv_after_body(x) + res)

        return x, w




class Downsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Downsample, self).__init__()
        self.ratio = ratio
        dconvs = []
        for i in range(int(np.log2(ratio))):
            dconvs.append(nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=1, dilation=1, groups=n_channels, bias=True))

        self.downsample = nn.Sequential(*dconvs)

    def forward(self,x):
        h = self.downsample(x)
        return h


class Upsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Upsample, self).__init__()
        uconvs = []
        for i in range(int(np.log2(ratio))):
            uconvs.append(nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.upsample = nn.Sequential(*uconvs)


    def forward(self,x):
        h = self.upsample(x)
        return h

class DCT(nn.Module):
    def __init__(self, n_colors, upscale_factor, n_feats=180):
        super(DCT, self).__init__()
        kernel_size = 3
        self.up_factor = upscale_factor
        Ch = n_colors
        self.conv = nn.Conv2d(Ch + 3, n_feats, kernel_size=3, stride=1, padding=1)
        self.headX = nn.Conv2d(n_colors, n_feats, kernel_size, stride=1, padding=3 // 2)
        self.headY = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(64, n_feats, kernel_size, stride=1, padding=3 // 2)
        )
        params = ListaParams(kernel_size=1, num_filters=160, stride=1, unfoldings=7)
        self.DA = ConvLista_T(params)
        self.pre_prior1 = Pre_Prior()
        self.pre_prior2 = Pre_Prior()
        self.pre_prior3 = Pre_Prior()
        self.body = dualTransformer(n_feats)

        self.fe_conv0 = torch.nn.Conv2d(in_channels=1*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv1 = torch.nn.Conv2d(in_channels=2*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=3*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)

        self.RT = nn.Sequential(nn.Conv2d(3, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.R = nn.Sequential(nn.Conv2d(Ch, 3, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        ## The modules for learning the measurement matrix B and B^T
        if self.up_factor == 8:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
            self.B = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=12, stride=8, padding=2), nn.LeakyReLU())
        elif self.up_factor == 16:
            self.BT = nn.Sequential(nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                    nn.LeakyReLU(),
                                    nn.ConvTranspose2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())
            self.B = nn.Sequential(nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(Ch, Ch, kernel_size=6, stride=4, padding=1), nn.LeakyReLU())

        self.lamda_0 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_1 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_2 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.lamda_3 = Parameter(torch.ones(1), requires_grad=True)
        self.eta_3 = Parameter(torch.ones(1), requires_grad=True)


        self.final = nn.Conv2d(n_feats, n_colors, kernel_size, stride=1, padding=3 // 2)

    def reconnect(self, Res1, Res2, Xt, Ut, Wt, DAt, i):
        if i == 0:
            eta = self.eta_0
            lamda = self.lamda_0
        elif i == 1:
            eta = self.eta_1
            lamda = self.lamda_1
        elif i == 2:
            eta = self.eta_2
            lamda = self.lamda_2
        elif i == 3:
            eta = self.eta_3
            lamda = self.lamda_3
        # elif i == 4:
        #     eta = self.eta_4
        #     lamda = self.lamda_4
        # elif i == 5:
        #     eta = self.eta_5
        #     lamda = self.lamda_5

        Xt = Xt - 2 * eta * (Res1 + Res2 + Wt * (Xt - Ut) + lamda * (Xt - DAt))
        return Xt

    def forward(self, x, y, guided_fea):
        Zt0 = torch.nn.functional.interpolate(x, scale_factor=self.up_factor, mode='bicubic', align_corners=False)

        DAt0 = self.DA(Zt0)
        ZtR = self.R(Zt0)
        Res1 = self.RT(ZtR - y)
        BZt = self.B(Zt0)
        Res2 = self.BT(BZt - x)
        Y = self.headY(y)
        Zt_input0 = self.fe_conv0(self.conv(torch.cat((Zt0, y), 1)))
        Ut0, Wt0 = self.body(Zt_input0,Y, self.pre_prior1(guided_fea))
        Zt1 = self.reconnect(Res1, Res2, Zt0, Ut0, Wt0, DAt0, 0)

        DAt1 = self.DA(Zt1)
        ZtR = self.R(Zt1)
        Res1 = self.RT(ZtR - y)
        BZt = self.B(Zt1)
        Res2 = self.BT(BZt - x)
        Zt_input1 = self.fe_conv1(torch.cat([Zt_input0, self.conv(torch.cat((Zt1, y), 1))], 1))
        Ut1, Wt1 = self.body(Zt_input1,Y, self.pre_prior2(guided_fea))
        Zt2 = self.reconnect(Res1, Res2, Zt1, Ut1, Wt1, DAt1, 1)

        DAt2 = self.DA(Zt2)
        ZtR = self.R(Zt2)
        Res1 = self.RT(ZtR - y)
        BZt = self.B(Zt2)
        Res2 = self.BT(BZt - x)
        Zt_input2 = self.fe_conv2(torch.cat([Zt_input0, Zt_input1, self.conv(torch.cat((Zt2, y), 1))], 1))
        Ut2, Wt2 = self.body(Zt_input2,Y, self.pre_prior3(guided_fea))
        Zt3 = self.reconnect(Res1, Res2, Zt2, Ut2, Wt2, DAt2, 2)
        return Zt3


class Pre_Prior(nn.Module):
    def __init__(self, n_feats=512):
        super(Pre_Prior, self).__init__()
        # kernel_size = 3
        # self.head = nn.Conv2d(512, 128, kernel_size, padding=3 // 2)
        # self.feture = nn.Conv2d(128, n_feats, kernel_size, padding=3 // 2)
        # self.body = swin_Transformer(n_feats)
        # self.upsample = Upsampler(conv, up_scale, n_feats)

        self.cha = nn.Sequential(nn.Linear(256, 256, bias=False),
                                 nn.Linear(256, 256, bias=False))
        # self.spa = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.ReLU(),
        #                          nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(),
        #                          nn.Conv2d(128, 512, 1, 1, 0)
        #                          )

        # self.conv3d1 = wn(nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        # self.conv3d2 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))
        # self.conv3d3 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        cha = x
        # spa = self.spa(spa)+spa
        cha = self.cha(torch.squeeze(torch.squeeze(cha, dim=-1), dim=-1)).unsqueeze(-1).unsqueeze(-1)

        return cha


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim=256, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # torch.nn.init.uniform_(self.embeddings.weight, 0, 3)
        torch.nn.init.uniform_(self.embeddings.weight, -1.0/self.num_embeddings, 1.0/self.num_embeddings)
        # torch.nn.init.orthogonal_(self.embeddings.weight)

    def forward(self, x, target=None):
        B, C, H, W = x.shape
        encoding_indices = self.get_code_indices(x, target)
        quantized0 = self.quantize(encoding_indices)
        quantized = quantized0.view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # weight, encoding_indices = self.get_code_indices(x)
        # quantized = self.quantize(weight, encoding_indices)

        if not self.training:
            return quantized, encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print("??????????????????",loss)

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, flat_x, target=None):
        # flag = self.training
        flat_x = flat_x.permute(0, 2, 3, 1).contiguous()
        z_flattened = flat_x.view(-1, self.embedding_dim)
        # z_flattened = F.normalize(z_flattened, p=2, dim=1)
        weight = self.embeddings.weight
        # weight = F.normalize(weight, p=2, dim=1)
        flag = False
        if flag:
            # print(target.dtype)
            # raise ValueError("target type error! ")
            encoding_indices = target
        else:
            # compute L2 distance
            distances = (
                    torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                    torch.sum(weight ** 2, dim=1) -
                    2. * torch.matmul(z_flattened, weight.t())
            )  # [N, M]
            # dis, encoding_indices = distances.topk(k=10)
            # index = F.gumbel_softmax(distances, tau=1, hard=False)
            # encoding_indices = torch.argmin(index, dim=1)  # [N,]
            encoding_indices = torch.argmin(distances, dim=1)  # [N,]
            # weight = F.softmax(dis / 2, dim=1)
        return encoding_indices
        # return weight, encoding_indices

    # def quantize(self, weight, encoding_indices):
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        # b, k = weight.size()
        # self.embeddings(encoding_indices)
        # quantized = torch.stack(
        #     [torch.index_select(input=self.embeddings.weight, dim=0, index=encoding_indices[i, :]) for i in range(b)])
        # weight = weight.view(b, 1, k).contiguous()
        # quantized = torch.bmm(weight, quantized).view(b, -1).contiguous()
        # return quantized
        return self.embeddings(encoding_indices)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class CPEN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6):
        super(CPEN, self).__init__()
        # self.scale = scale
        # if scale == 2:
        E1 = [nn.Conv2d(64, n_feats, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.1, True)]
        # elif scale == 1:
        #     E1 = [nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
        #           nn.LeakyReLU(0.1, True)]
        # else:
        #     E1 = [nn.Conv2d(64, n_feats, kernel_size=3, padding=1),
        #           nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            # nn.AdaptiveAvgPool2d(4),
            # nn.MaxPool2d(2),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
            nn.LeakyReLU(0.1, True)
        )

        # self.mlp2 = nn.Sequential(
        #     nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
        #     nn.LeakyReLU(0.1, True)
        # )
        self.VQ = VectorQuantizer(n_feats * 4, 512, 0.25)
        # self.pixel_unshuffle = nn.PixelUnshuffle(4)
        # self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x, gt=False):
        # gt0 = self.pixel_unshuffle(gt)
        # if self.scale == 2:
        #     feat = self.pixel_unshufflev2(x)
        # elif self.scale == 1:
        #     feat = self.pixel_unshuffle(x)
        # else:
        #     feat = x
        # fea = x
        # x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        gt_fea = fea1
        # fea1 = self.mlp2(fea1)
        if self.training:
            fea2, loss, idx = self.VQ(fea1)
            if gt:
                return fea2, gt_fea, loss, idx
            else:
                return fea2, fea1, loss, idx

        else:
            fea2, idx = self.VQ(fea1)
            return fea2, fea1, idx

class Decoding(nn.Module):
    def __init__(self, feat=64, dim=48, scale=1):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.scale = scale
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=feat, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D1_ending = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=dim*8, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D2_ending = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64*2, out_channels=dim*4, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        # self.D3_ending = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=64*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        # self.D4_ending = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=32, out_channels=dim, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        # self.D4_ending1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=32*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        self.D5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.ending = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=31, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, E):
        ## decoding blocks
        D1 = self.D2(F.interpolate(self.D1(E), scale_factor=2, mode=self.upMode))
        D2 = self.D3(F.interpolate(D1, scale_factor=2, mode=self.upMode))
        D3 = self.D4(F.interpolate(D2, scale_factor=2, mode=self.upMode))
        if self.scale == 2:
            D4 = self.D5(F.interpolate(D3, scale_factor=2, mode=self.upMode))
            return self.ending(D4)
        elif self.scale == 3:
            D4 = self.D5(F.interpolate(D3, scale_factor=3, mode=self.upMode))
            return self.ending(D4)
        elif self.scale == 4:
            D4 = self.D5(F.interpolate(D3, scale_factor=2, mode=self.upMode))
            D5 = self.D6(F.interpolate(D4, scale_factor=2, mode=self.upMode))
            return self.ending(D5)
        else:
            D4 = self.D5(D3)
            D5 = self.D6(D4)
            return self.ending(D5)

class DGSMPIR(nn.Module):

    def __init__(self, ChDim, pscale_factor):
        super(DGSMPIR, self).__init__()
        self.G = DCT(ChDim, pscale_factor)

        self.E_pre_gt = nn.Sequential(nn.Conv2d(in_channels=ChDim, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        Res = [
            ResBlock(
                default_conv, 64, kernel_size=3
            ) for _ in range(8)
        ]
        self.E_pre_lr = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                      *Res)
        self.E = CPEN(n_feats=64, n_encoder_res=6)
        self.L1 = nn.L1Loss()
        self.pool = nn.Sequential(nn.Conv2d(in_channels=256*2, out_channels=64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0),
                                  nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)
        #
        # torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)

        self.reset_parameters()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.xavier_normal_(m.weight.data)
    #         nn.init.constant_(m.bias.data, 0.0)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size*4 - h % (self.window_size*4)) % (self.window_size*4)
        mod_pad_w = (self.window_size*4 - w % (self.window_size*4)) % (self.window_size*4)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w = x.size()
        return x

    def forward(self, x, rgb, gt):
        if self.training:
            _, _, H, W = gt.shape
            F_rgb = self.E_pre_lr(rgb)
            F_gt = self.E_pre_gt(gt)
            IPRS1, S1_IPR, vq_loss, vq_idx = self.E(F_rgb)  # IPRS1 [B, 256]
            GTIPRS, GT_IPR, vq_loss1, vq_idx1 = self.E(F_gt, gt=True)
            # rec = self.D(GTIPRS)
            # rec2 = self.D(IPRS1)
            # _, up8_1, up8, up4, up2, up0 = self.D(IPRS1)
            rec_loss = self.L1(GT_IPR, S1_IPR)
            cat = torch.cat([IPRS1, S1_IPR], dim=1)
            # cat = self.BAM(torch.cat([IPRS1, S1_IPR], dim=1))
            sr = self.G(x, rgb, self.pool(cat))

            return sr
        else:
            IPRS1, S1_IPR, vq_idx = self.E(self.E_pre_lr(rgb))
            cat = torch.cat([IPRS1, S1_IPR], dim=1)
            # cat = self.BAM(torch.cat([IPRS1, S1_IPR], dim=1))
            sr = self.G(x, rgb, self.pool(cat))

            return sr

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops

