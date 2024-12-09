from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
from .customized_linear import CustomizedLinear
from einops import rearrange
import random
import numpy as np
import pandas as pd
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")






class my_Conv1d(nn.Module):
    def __init__(self):
        super(my_Conv1d, self).__init__()
        # self.kernel = torch.randn(48, 1, 10).to('cuda')
        # self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
        # self.weight = nn.Parameter(torch.Tensor(48, 1, 10))

        # self.conv1 = nn.Conv1d(1, 6, 5, padding=1, stride=3)
        # self.conv2 = nn.Conv1d(6, 48, 5, padding=1, stride=3)
        # self.conv1 = self.conv1.to('cuda')
        # self.conv2 = self.conv2.to('cuda')


        self.conv1 = nn.Conv1d(1, 48, 10, padding=0, stride=10)
        self.conv1 = self.conv1.to('cuda')


    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], 1, 3000))
        x = self.conv1(x)  # 8*48*300
        x = x.permute(0, 2, 1)

        # x = torch.reshape(x, (x.shape[0], 1, 3000))
        # x = self.conv1(x)
        # x = self.conv2(x)    # 8*48*333
        # x = x.permute(0, 2, 1)

        # x = torch.reshape(x, (x.shape[0], 1, 3000))
        # x = torch.nn.functional.conv1d(x, weight=self.kernel, bias=None, stride=10, padding=0, dilation=1, groups=1)
        # x = x.permute(0, 2, 1)
        return x









def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): length of gene

    Returns:
        x: (B, L, C)
    """
    B_, W, C = windows.size()
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, C, L // window_size, window_size)
    return x

def window_partition(x, window_size):
    """
        Args:
            x: (B, L, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L//window_size, window_size,C)
    B, W, S, C = x.shape
    windows = x.reshape(B*W, S, C)
    return windows

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
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
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class SwinTransformerBlock(nn.Module):
    """
    one-dimensional Swin Transformer block.
    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=10, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution, self.window_size)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.fused_window_process = fused_window_process


    def forward(self, x):
        W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=self.shift_size, dims=1)
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, S, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C   240*10*48

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, L)
            x = torch.roll(shifted_x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, L)
            x = shifted_x

        x = x.view(B, -1, C)
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic one-dimensional Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x



class PatchEmbed(nn.Module):
    """
    gene to patch_embedding
    
    Args:
        gene_len (int):  Gene length of individual cells.  Default: 3000.
        patch_size (int): Patch token size. Default: 10.
        in_channels (int): Number of input gene channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    
    def __init__(self, gene_len, patch_size, token, in_channels=1, embed_dim=48, norm_layer=None):
        super().__init__()
        # if gene_len == patch_size * token:
        #     patches_resolution = gene_len // patch_size
        # else:
        #     patches_resolution = gene_len // patch_size

        self.gene_len = gene_len
        self.patch_size = patch_size
        self.patches_resolution = token
        self.num_patches = token
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_channels=self.in_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)    # B*C*P
        x = x.permute(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x



class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        input_resolution = self.input_resolution
        B, L, C = x.shape
        assert input_resolution == L, "input feature has wrong size"
        x = x.view(B, L, C)

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)

        return x



def Padding(x, patch_size, token,device):
    B, C, L = x.shape
    if patch_size * token == L:
        return x
    else:
        zeros = torch.tensor(torch.zeros(B, C, patch_size * token - L).to(device))  # Zero padding
        return torch.cat((x, zeros), dim=2)




class SwinTransformer(nn.Module):
    """
        one-dimensional SwinTransformer
    Args:
        gene_len (int):  Gene length of individual cells.  Default: 3000.
        patch_size (int): Patch token size. Default: 10.
        in_channels (int): Number of input gene channels. Default: 1
        num_classes (int): Number of classes for classification head. Default: 13
        embed_dim (int): Patch embedding dimension. Default: 48
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 10
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    
    def __init__(self, num_genes, token=200, in_channels=1, num_classes=13,
                 embed_dim=48, depths=None, num_heads=None, window_size=10,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,device='cuda:0',
                 ape=False, patch_norm=True, use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()
        if depths is None:
            # depths = [2, 2, 6, 2]
            depths = [2, 6]
            # depths = [8]
        if num_heads is None:
            # num_heads = [2, 4, 8, 16]
            num_heads = [2, 3]
            # num_heads = [4]

        self.device=device
        self.token = token
        if num_genes % token == 0:
            self.patch_size = num_genes // token
        else:
            self.patch_size = num_genes // token + 1
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split genes into non-overlapping patches
        self.patch_embed = PatchEmbed(gene_len=num_genes, patch_size=self.patch_size, token=self.token, in_channels=in_channels,
                                      embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule0

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=patches_resolution // (2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.token*self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        classifier_hidden_dim = int(embed_dim * mlp_ratio)
        self.classifier = Classifier(in_features=self.token*self.embed_dim, hidden_features=classifier_hidden_dim,
                                     out_features=num_classes, drop=drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = Padding(x, self.patch_size, self.token,device=self.device)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        # x = x.permute(0, 2, 1)
        return x


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = self.forward_features(x)
        x = torch.reshape(x, [x.shape[0], -1])
        # x = self.head(x)  # classify
        x = self.classifier(x)
        return x














def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


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

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        hhh, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(hhh)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights

def get_weight(att_mat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v


class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask, fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None, max_gs=200,
                 act_layer=None):
        """
        Args:,   embed_layer=FeatureEmbed    embed_layer=my_Conv1d
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData) 
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate 
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.swin_transformer = SwinTransformer(num_genes=num_genes, token=max_gs)
        # self.feature_embed = embed_layer()
        # self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        #self.blocks = nn.Sequential(*[
        #    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #          norm_layer=norm_layer, act_layer=act_layer)
        #    for i in range(depth)
        #])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
    def forward_features(self, x):
        # x = self.feature_embed(x)

        x = self.swin_transformer(x)

        # x = torch.reshape(x, (x.shape[0], 1, 3000))
        # kernel = torch.randn(48, 1, 10).to('cuda')
        # # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # weight = nn.Parameter(data=kernel, requires_grad=False)
        # x = torch.nn.functional.conv1d(x,weight,bias=None, stride=10, padding=0, dilation=1, groups=1)
        # x = x.permute(0, 2, 1)

        # conv1 = nn.Conv1d(1, 48, 10, padding=0, stride=10)
        # conv1 = conv1.to('cuda')
        # x = torch.reshape(x, (x.shape[0], 1, 3000))
        # x = conv1(x)  # 8*48*300
        # x = x.permute(0, 2, 1)

        # conv1 = nn.Conv1d(1, 6, 5, padding=1, stride=3)
        # conv2 = nn.Conv1d(6, 48, 5, padding=1, stride=3)
        # conv1 = conv1.to('cuda')
        # conv2 = conv2.to('cuda')
        # x = torch.reshape(x, (x.shape[0], 1, 3000))
        # x = conv1(x)
        # x = conv2(x)    # 8*48*333
        # x = x.permute(0, 2, 1)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:  # ViT中就是None
            x = torch.cat((cls_token, x), dim=1) 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]),attn_weights 
        else:
            return x[:, 0], x[:, 1],attn_weights
    def forward(self, x):
        latent, attn_weights = self.forward_features(x)

        if self.head_dist is not None: 
            latent, latent_dist = self.head(latent[0]), self.head_dist(latent[1])
            if self.training and not torch.jit.is_scripting():
                return latent, latent_dist
            else:
                return (latent+latent_dist) / 2
        else:
            pre = self.head(latent) 
        return latent, pre, attn_weights

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  

def scTrans_model(num_classes, num_genes, mask, embed_dim=48,depth=2,num_heads=4,max_gs=200,has_logits: bool = True):
    model = Transformer(num_classes=num_classes, 
                        num_genes=num_genes, 
                        mask = mask,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,max_gs=max_gs,
                        representation_size=embed_dim if has_logits else None)
    return model

