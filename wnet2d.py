# nnUNetTrainer_WNet2D.py
# Clean WNet2D core with multi-scale local blocks + global blocks + GSB-Mamba, ready for import.
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.modules.mamba_simple import Mamba

BNNorm2d = nn.BatchNorm2d
LNNorm = nn.LayerNorm
Activation = nn.GELU

def count_parameters(model, only_trainable=True):
    params = filter(lambda p: p.requires_grad, model.parameters()) if only_trainable else model.parameters()
    return sum(p.numel() for p in params)

# --------------------------- Local multi-scale block ---------------------------
class MultiScaleLSBv2(nn.Module):
    def __init__(self, inplanes, planes=None, groups=1):
        super().__init__()
        planes = inplanes if planes is None else planes
        self.conv3 = nn.Conv2d(inplanes, inplanes, 3, padding=2, dilation=2, groups=inplanes)
        self.conv5 = nn.Conv2d(inplanes, inplanes, 5, padding=4, dilation=2, groups=inplanes)
        self.conv7 = nn.Conv2d(inplanes, inplanes, 7, padding=6, dilation=2, groups=inplanes)
        self.fuse = nn.Sequential(
            nn.Conv2d(inplanes * 3, planes, 1, bias=False),
            BNNorm2d(planes),
            Activation()
        )
        self.skip_proj = nn.Identity() if inplanes == planes else nn.Conv2d(inplanes, planes, 1, bias=False)

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        out = self.fuse(torch.cat([f3, f5, f7], dim=1))
        return out + self.skip_proj(x)

# --------------------------- Utilities ---------------------------
class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    def forward(self, x): return self.pool(x) - x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

# --------------------------- Global block (pooled multi-branch + MLP) ---------------------------
class global_block(nn.Module):
    def __init__(self, inplanes, planes=None, pool_sizes=(3,5,7), mlp_ratio=4.0, drop=0.0, drop_path=0.0, num_heads=None, sr_ratio=None):
        super().__init__()
        planes = inplanes if planes is None else planes
        self.pool3 = nn.AvgPool2d(pool_sizes[0], stride=1, padding=pool_sizes[0]//2)
        self.pool5 = nn.AvgPool2d(pool_sizes[1], stride=1, padding=pool_sizes[1]//2)
        self.pool7 = nn.AvgPool2d(pool_sizes[2], stride=1, padding=pool_sizes[2]//2)

        self.conv3 = nn.Conv2d(inplanes, inplanes, 1)
        self.conv5 = nn.Conv2d(inplanes, inplanes, 1)
        self.conv7 = nn.Conv2d(inplanes, inplanes, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(inplanes * 3, planes, 1, bias=False),
            GroupNorm(planes),
            Activation()
        )

        hidden_dim = int(planes * mlp_ratio)
        self.mlp = Mlp(planes, hidden_dim, planes, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_proj = nn.Identity() if inplanes == planes else nn.Conv2d(inplanes, planes, 1, bias=False)

    def forward(self, x):
        identity = self.skip_proj(x)
        f3 = self.conv3(self.pool3(x))
        f5 = self.conv5(self.pool5(x))
        f7 = self.conv7(self.pool7(x))
        out = self.fuse(torch.cat([f3, f5, f7], dim=1))
        out = out + self.drop_path(self.mlp(out))
        return out + identity

# --------------------------- State-space (Mamba) block stack ---------------------------
def build_2d_sincos_position_embedding(h, w, dim, temperature=10000.):
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    assert dim % 4 == 0
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1. / (temperature ** omega)
    out_x = grid_x.flatten()[:, None] * omega[None, :]
    out_y = grid_y.flatten()[:, None] * omega[None, :]
    pe = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
    pe = pe.view(h, w, dim).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    return pe

class MambaLayer(nn.Module):
    def __init__(self, dim, seq_len=0):
        super().__init__()
        self.linear_proj = nn.Conv2d(dim, dim, 1)
        self.mamba = Mamba(d_model=dim)
        self.seq_len = seq_len
    def forward(self, x):
        B, C, H, W = x.size()
        x = self.linear_proj(x)
        x = x.flatten(2).transpose(1, 2)   # [B,HW,C]
        x = self.mamba(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class GSB_Mamba(nn.Module):
    def __init__(self, in_dim=192, out_dim=64, depth=1, seq_hw=None):
        super().__init__()
        self.proj_in = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            MambaLayer(out_dim, seq_len=(seq_hw * seq_hw) if seq_hw is not None else 0)
            for _ in range(depth)
        ])
        self.pe = None

    def _maybe_build_pe(self, x):
        B, C, H, W = x.shape
        if (self.pe is None) or (self.pe.shape[1:] != (C, H, W)):
            self.pe = build_2d_sincos_position_embedding(H, W, C).to(x.device)

    def forward(self, x):
        x = self.proj_in(x)
        self._maybe_build_pe(x)
        h = x + self.pe
        for block in self.blocks:
            h = block(h) + h
        return h

# --------------------------- Down/Up sampling & local wrappers ---------------------------
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )
    def forward(self, x): return self.up(x)

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1, bias=False),
            BNNorm2d(ch_out),
            Activation()
        )
    def forward(self, x): return self.down(x)

class OPE(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, 1, 1)
        self.bn1 = BNNorm2d(inplanes)
        self.act = Activation()
        self.down = down_conv(inplanes, planes)
    def forward(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.act(out)
        out = self.down(out)
        return out

class local_block(nn.Module):
    def __init__(self, inplanes, hidden_planes, planes, groups=1, down_or_up=None):
        super().__init__()
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                MultiScaleLSBv2(inplanes=inplanes, planes=hidden_planes, groups=groups),
            )
        elif down_or_up == 'down':
            self.BasicBlock = nn.Sequential(
                MultiScaleLSBv2(inplanes=inplanes, planes=hidden_planes, groups=groups),
                down_conv(hidden_planes, planes)
            )
        elif down_or_up == 'up':
            self.BasicBlock = nn.Sequential(
                MultiScaleLSBv2(inplanes=inplanes, planes=hidden_planes, groups=groups),
                up_conv(hidden_planes, planes),
            )
        else:
            raise ValueError("down_or_up must be one of {None,'down','up'}")
    def forward(self, x): return self.BasicBlock(x)

# --------------------------- WNet2D (W-shaped dual U) ---------------------------
class WNet2D(nn.Module):
    def __init__(self, in_channel=3, num_classes=2, deep_supervised=False,
                 layer_channel=[16, 32, 64, 128, 256], global_dim=[8, 16, 32, 64, 128],
                 num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1]):
        super().__init__()
        self.deep_supervised = deep_supervised

        # encoder 1
        self.input_l0 = nn.Sequential(
            nn.Conv2d(in_channel, layer_channel[0], 3, 1, 1),
            BNNorm2d(layer_channel[0]),
            Activation(),
            nn.Conv2d(layer_channel[0], layer_channel[0], 3, 1, 1),
            BNNorm2d(layer_channel[0]),
            Activation()
        )
        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(layer_channel[0], global_dim[0])

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(layer_channel[1], global_dim[1])

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(layer_channel[2], global_dim[2])

        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(layer_channel[3], global_dim[3])

        # decoder 1
        self.decoder1_l4_local = local_block(layer_channel[4], layer_channel[4], layer_channel[3], down_or_up='up')
        self.decoder1_l4_global = global_block(layer_channel[4], global_dim[4])

        self.decoder1_l3_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')
        self.decoder1_l3_global = global_block(layer_channel[3] + global_dim[3], global_dim[3])

        self.decoder1_l2_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')
        self.decoder1_l2_global = global_block(layer_channel[2] + global_dim[2], global_dim[2])

        self.decoder1_l1_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')
        self.decoder1_l1_global = global_block(layer_channel[1] + global_dim[1], global_dim[1])

        # encoder 2
        self.encoder2_l1_local = local_block(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[1], down_or_up='down')
        self.encoder2_l1_global = global_block(layer_channel[0] + global_dim[0], global_dim[0])

        self.encoder2_l2_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[2], down_or_up='down')
        self.encoder2_l2_global = global_block(layer_channel[1] + global_dim[1], global_dim[1])

        self.encoder2_l3_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[3], down_or_up='down')
        self.encoder2_l3_global = global_block(layer_channel[2] + global_dim[2], global_dim[2])

        self.encoder2_l4_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[4], down_or_up='down')
        self.encoder2_l4_global = GSB_Mamba(in_dim=layer_channel[3] + global_dim[3], out_dim=global_dim[3], depth=3, seq_hw=64)

        # decoder 2 + deep supervision heads
        self.decoder2_l4_local_output = nn.Conv2d(layer_channel[4], num_classes, 1, 1, 0)
        self.decoder2_l4_local = local_block(layer_channel[4] + global_dim[4], layer_channel[4], layer_channel[3], down_or_up='up')

        self.decoder2_l3_local_output = nn.Conv2d(layer_channel[3], num_classes, 1, 1, 0)
        self.decoder2_l3_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')

        self.decoder2_l2_local_output = nn.Conv2d(layer_channel[2], num_classes, 1, 1, 0)
        self.decoder2_l2_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')

        self.decoder2_l1_local_output = nn.Conv2d(layer_channel[1], num_classes, 1, 1, 0)
        self.decoder2_l1_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')

        self.output_l0 = nn.Sequential(
            local_block(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[0], down_or_up=None),
            nn.Conv2d(layer_channel[0], num_classes, 1, 1, 0)
        )

    def forward(self, x, return_feats: bool=False, verbose: bool=False):
        feats = {}
        outputs = []

        def maybe_log(name, t):
            if verbose:
                print(f"[{name}] shape={tuple(t.shape)}")
            if return_feats and name in {"input","x_e1_l3_global","encoder2_l4_global","x_d2_l3_local","final"}:
                feats[name] = t

        maybe_log("input", x)

        # encoder-decoder 1
        x_e1_l0 = self.input_l0(x)
        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0); x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)
        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local); x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)
        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local); x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)
        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local); x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local); x_d1_l4_global = self.decoder1_l4_global(x_e1_l4_local)
        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)

        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3); x_d1_l3_global = self.decoder1_l3_global(x_d1_l3)
        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)

        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2); x_d1_l2_global = self.decoder1_l2_global(x_d1_l2)
        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)

        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1); x_d1_l1_global = self.decoder1_l1_global(x_d1_l1)

        # encoder-decoder 2
        x_e2_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        x_e2_l1_local = self.encoder2_l1_local(x_e2_l0); x_e2_l0_global = self.encoder2_l1_global(x_e2_l0)

        x_e2_l1 = torch.cat((x_e2_l1_local, x_d1_l1_global), dim=1)
        x_e2_l2_local = self.encoder2_l2_local(x_e2_l1); x_e2_l1_global = self.encoder2_l2_global(x_e2_l1)

        x_e2_l2 = torch.cat((x_e2_l2_local, x_d1_l2_global), dim=1)
        x_e2_l3_local = self.encoder2_l3_local(x_e2_l2); x_e2_l2_global = self.encoder2_l3_global(x_e2_l2)

        x_e2_l3 = torch.cat((x_e2_l3_local, x_d1_l3_global), dim=1)
        x_e2_l4_local = self.encoder2_l4_local(x_e2_l3); x_e2_l3_global = self.encoder2_l4_global(x_e2_l3)

        out4 = self.decoder2_l4_local_output(x_e2_l4_local); outputs.append(out4)
        x_e2_l4 = torch.cat((x_e2_l4_local, x_d1_l4_global), dim=1)
        x_d2_l3_local = self.decoder2_l4_local(x_e2_l4)

        out3 = self.decoder2_l3_local_output(x_d2_l3_local); outputs.append(out3)
        x_d2_l3 = torch.cat((x_d2_l3_local, x_e2_l3_global), dim=1)
        x_d2_l2_local = self.decoder2_l3_local(x_d2_l3)

        out2 = self.decoder2_l2_local_output(x_d2_l2_local); outputs.append(out2)
        x_d2_l2 = torch.cat((x_d2_l2_local, x_e2_l2_global), dim=1)
        x_d2_l1_local = self.decoder2_l2_local(x_d2_l2)

        out1 = self.decoder2_l1_local_output(x_d2_l1_local); outputs.append(out1)
        x_d2_l1 = torch.cat((x_d2_l1_local, x_e2_l1_global), dim=1)
        x_d2_l0_local = self.decoder2_l1_local(x_d2_l1)

        x_d2_l0 = torch.cat((x_d2_l0_local, x_e2_l0_global), dim=1)
        final_out = self.output_l0(x_d2_l0); outputs.append(final_out)
        maybe_log("final", final_out)

        if self.deep_supervised:
            return outputs[::-1], feats if return_feats else outputs[::-1]
        else:
            return (final_out, feats) if return_feats else final_out

def build_wnet2d(in_channels=3, num_classes=2, deep_supervised=False):
    return WNet2D(in_channel=in_channels, num_classes=num_classes, deep_supervised=deep_supervised)
