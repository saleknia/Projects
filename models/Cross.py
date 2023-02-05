import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops
import timm
from torchvision import models as resnet_model
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple

class Cross(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
    
        self.encoder = DAT(
                            img_size=224,
                            patch_size=4,
                            num_classes=1000,
                            expansion=4,
                            dim_stem=96,
                            dims=[96, 192, 384, 768],
                            depths=[2, 2, 6, 2],
                            stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
                            heads=[3, 6, 12, 24],
                            window_sizes=[7, 7, 7, 7] ,
                            groups=[-1, -1, 3, 6],
                            use_pes=[False, False, True, True],
                            dwc_pes=[False, False, False, False],
                            strides=[-1, -1, 1, 1],
                            sr_ratios=[-1, -1, -1, -1],
                            offset_range_factor=[-1, -1, 2, 2],
                            no_offs=[False, False, False, False],
                            fixed_pes=[False, False, False, False],
                            use_dwc_mlps=[False, False, False, False],
                            use_conv_patches=False,
                            drop_rate=0.0,
                            attn_drop_rate=0.0,
                            drop_path_rate=0.2,
                        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=768, out_features=512, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=256, out_features=40, bias=True),
        )

    def forward(self, x):
        x_input = x.float()
        B, C, H, W = x.shape
        x = self.encoder(x_input)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
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
        return f'dim={self.dim}, group_size={self.group_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 group with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops


class CrossFormerBlock(nn.Module):
    r""" CrossFormer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, group_size=7, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        if min(self.input_resolution) <= self.group_size:
            # if group size is larger than input resolution, we don't partition groups
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # group embeddings
        G = self.group_size
        if self.lsda_flag == 0: # 0 for SDA
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
        else: # 1 for LDA
            x = x.reshape(B, G, H // G, G, W // G, C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * H * W // G**2, G**2, C)

        # multi-head self-attention
        x = self.attn(x, mask=self.attn_mask)  # nW*B, G*G, C

        # ungroup embeddings
        x = x.reshape(B, H // G, W // G, G, G, C)
        if self.lsda_flag == 0:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # LSDA
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
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

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=[2], num_input_patch_size=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x).flatten(2).transpose(1, 2)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                out_dim = 2 * self.dim // 2 ** i
            else:
                out_dim = 2 * self.dim // 2 ** (i + 1)
            flops += (H // 2) * (W // 2) * ps * ps * out_dim * self.dim
        return flops


class Stage(nn.Module):
    """ CrossFormer blocks for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): variable G in the paper, one group has GxG embeddings
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

    def __init__(self, dim, input_resolution, depth, num_heads, group_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            self.blocks.append(CrossFormerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, group_size=group_size,
                                 lsda_flag=lsda_flag,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 num_patch_size=num_patch_size))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, 
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
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


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class CrossFormer(nn.Module):
    r""" CrossFormer
        A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
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
    """


    def __init__(self, img_size=224, patch_size=[4], in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 group_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, merge_size=[[2], [2], [2]], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               group_size=group_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               patch_size_end=patch_size_end,
                               num_patch_size=num_patch_size)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        
        checkpoint = torch.load('/content/drive/MyDrive/crossformer-s.pth', map_location='cpu')
        state_dict = checkpoint['model']
        self.load_state_dict(state_dict, strict=False)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import einops
import timm
from torchvision import models as resnet_model
from .CTrans import ChannelTransformer
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple


class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

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

    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        
        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C
        
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total) # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn =(b * nW) h w w
            # mask =nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x)) # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W
        
        return x, None, None

class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)
      
    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None
    

class DAttentionBaseline(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, stage_idx
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
    
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        if self.no_off:
            offset = offset.fill(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        if self.use_pe:
            
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        
        y = self.proj_drop(self.proj_out(out))
        
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec={stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references

class DAT(nn.Module):

    def __init__(self, img_size=448, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], 
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 pretrain=True,
                 **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.reset_parameters()
        if pretrain:
            checkpoint = torch.load('/content/drive/MyDrive/dat_tiny_in1k_224.pth', map_location='cpu') 
            state_dict = checkpoint['model']
            self.load_pretrained(state_dict)

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def forward(self, x):
        
        x = self.patch_proj(x)
        positions = []
        references = []
        for i in range(3):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        # x = self.cls_head(x)
        
        return x




