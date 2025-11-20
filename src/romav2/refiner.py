from functools import partial

from torch.nn import functional as F
from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
from romav2.local_correlation import local_correlation
from romav2.geometry import get_normalized_grid
from romav2.device import device
from romav2.geometry import bhwc_grid_sample
from romav2.types import RefinersType, NormType


def create_block(
    in_dim: int,
    out_dim: int,
    kernel_size: int,
    bn_momentum: float,
    norm_type_name: NormType = "batch",
):
    num_groups = in_dim
    assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
    conv_depthwise = nn.Conv2d(
        in_dim,
        out_dim,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        groups=num_groups,
    )
    if norm_type_name == "batch":
        norm = nn.BatchNorm2d(out_dim, momentum=bn_momentum)
    else:
        raise TypeError(f"Unknown norm type: {norm_type_name}")
    relu = nn.ReLU(inplace=True)
    # conv_pointwise = nn.Linear(out_dim, out_dim)
    conv_pointwise = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
    return [conv_depthwise, norm, relu, conv_pointwise]


class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        bn_momentum: float,
        enable_amp: bool,
        norm_type_name: NormType = "batch",
    ):
        super().__init__()
        self.conv_depthwise, self.norm, self.relu, self.conv_pointwise = create_block(
            in_dim, out_dim, kernel_size, bn_momentum, norm_type_name
        )
        self.enable_amp = enable_amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=device.type, enabled=self.enable_amp, dtype=torch.bfloat16
        ):
            x = self.conv_depthwise(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.conv_pointwise(x)
        return x


class ConvRefiner(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        feat_dim: int
        proj_dim: int
        displacement_emb_dim: int
        local_corr_radius: int | None
        patch_size: int
        warp_dim: int = 2
        confidence_dim: int = 4
        kernel_size: int = 5
        hidden_blocks: int = 8
        norm_type_name: NormType = "batch"
        bn_momentum: float = 0.01
        enable_amp: bool = True
        refine_init: float = 4.0
        channels_last: bool = False
        block_type: Literal["roma"] = "roma"
        grid_sample_mode: Literal["bilinear", "bicubic"] = "bilinear"

    def __init__(
        self,
        cfg: Cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.feat_dim, cfg.proj_dim)
        hidden_dim = (
            2 * cfg.proj_dim
            + cfg.displacement_emb_dim
            + (
                (2 * cfg.local_corr_radius + 1) ** 2
                if cfg.local_corr_radius is not None
                else 0
            )
        )
        self.hidden_dim = hidden_dim
        if cfg.block_type == "roma":
            self.block1 = Block(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                norm_type_name=cfg.norm_type_name,
                bn_momentum=cfg.bn_momentum,
                kernel_size=cfg.kernel_size,
                enable_amp=cfg.enable_amp,
            )
            self.hidden_blocks = nn.Sequential(
                *[
                    Block(
                        hidden_dim,
                        hidden_dim,
                        norm_type_name=cfg.norm_type_name,
                        bn_momentum=cfg.bn_momentum,
                        kernel_size=cfg.kernel_size,
                        enable_amp=cfg.enable_amp,
                    )
                    for _ in range(cfg.hidden_blocks)
                ]
            )
        else:
            raise ValueError(f"Unknown block type: {cfg.block_type}")

        self.disp_emb = nn.Conv2d(2, cfg.displacement_emb_dim, 1, 1, 0)
        self.warp_head = nn.Conv2d(hidden_dim, cfg.warp_dim, 1, 1, 0)
        self.confidence_head = nn.Conv2d(hidden_dim, cfg.confidence_dim, 1, 1, 0)

    def forward(
        self,
        *,
        f_A: torch.Tensor,
        f_B: torch.Tensor,
        prev_warp: torch.Tensor,
        prev_confidence: torch.Tensor | None,
        scale_factor: torch.Tensor,
    ):
        B, H_A, W_A, _ = f_A.shape
        B, H_B, W_B, D = f_B.shape
        assert H_A == H_B and W_A == W_B, "Images must have the same height and width"
        prev_warp = prev_warp.detach()
        if prev_confidence is not None:
            prev_confidence = prev_confidence.detach()
        B, H_A, W_A, D = f_A.shape
        assert D == self.cfg.feat_dim, (
            f"Config feature dimension {self.cfg.feat_dim=} must be the same as the input feature dimension {D=}"
        )

        f_A = self.proj(f_A.reshape(B, H_A * W_A, self.cfg.feat_dim).float()).reshape(
            B, H_A, W_A, self.cfg.proj_dim
        )
        f_B = self.proj(f_B.reshape(B, H_B * W_B, self.cfg.feat_dim).float()).reshape(
            B, H_B, W_B, self.cfg.proj_dim
        )
        with torch.no_grad():
            f_BtoA = bhwc_grid_sample(
                f_B, prev_warp, mode=self.cfg.grid_sample_mode, align_corners=False
            )
        im_A_coords = get_normalized_grid(B, H_A, W_A)
        in_displacement = prev_warp - im_A_coords
        in_displacement_bdhw = in_displacement.permute(0, 3, 1, 2)
        emb_in_displacement = self.disp_emb(
            scale_factor[None, :, None, None] * in_displacement_bdhw
        )
        # Corr in other means take a kxk grid around the predicted coordinate in other image
        f_A_bdhw = f_A.permute(0, 3, 1, 2)
        f_B_bdhw = f_BtoA.permute(0, 3, 1, 2)
        d = torch.cat((f_A_bdhw, f_B_bdhw, emb_in_displacement), dim=1)
        if self.cfg.local_corr_radius is not None:
            local_corr = local_correlation(
                f_A_bdhw,
                f_B_bdhw,
                local_radius=self.cfg.local_corr_radius,
                warp=prev_warp,
                scale_factor=scale_factor,
            )
            d = torch.cat((d, local_corr), dim=1)
        # d = torch.cat((f_A_bdhw, f_B_bdhw, emb_in_displacement, local_corr), dim=1)
        if self.cfg.channels_last:
            d = d.to(memory_format=torch.channels_last)
        z = self.block1(d)
        z = self.hidden_blocks(z)
        z = z.float()

        displacement = self.warp_head(z)
        delta_confidence = self.confidence_head(z)
        displacement = displacement.permute(0, 2, 3, 1)
        delta_confidence = delta_confidence.permute(0, 2, 3, 1)
        displacement = displacement.view(B, H_A, W_A, self.cfg.warp_dim)
        delta_confidence = delta_confidence.view(B, H_A, W_A, self.cfg.confidence_dim)
        warp = prev_warp + displacement / (
            self.cfg.refine_init
            * torch.tensor((W_A, H_A), device=device)[None, None, None]
        )

        if delta_confidence.shape[-1] == 4:
            chol_eps = 1e-6
            l00 = F.softplus(delta_confidence[..., 1]) + chol_eps  # this is in pixels
            l10 = delta_confidence[..., 2]
            l11 = F.softplus(delta_confidence[..., 3]) + chol_eps
            p00 = l00 * l00
            p10 = l00 * l10
            p11 = l10 * l10 + l11 * l11
            delta_confidence = torch.stack(
                [delta_confidence[..., 0], p00, p10, p11], dim=-1
            )

        if prev_confidence is not None:
            if prev_confidence.shape[-1] == delta_confidence.shape[-1]:
                confidence = prev_confidence + delta_confidence
            else:
                padded_prev_confidence = torch.zeros_like(delta_confidence)
                padded_prev_confidence[..., :1] = prev_confidence
                confidence = padded_prev_confidence + delta_confidence
        else:
            confidence = delta_confidence

        return {"warp": warp, "confidence": confidence}


class Refiners:
    @dataclass(frozen=True)
    class Cfg:
        refiner_type: RefinersType = "roma-4-pow2"
        confidence_dim: int = 4
        grid_sample_mode: Literal["bilinear", "bicubic"] = "bilinear"

    def __new__(cls, cfg: Cfg):
        partial_refiner_coarse = partial(
            ConvRefiner.Cfg,
            confidence_dim=cfg.confidence_dim,
            grid_sample_mode=cfg.grid_sample_mode,
        )
        match cfg.refiner_type:
            case "roma-4-pow2":
                refiner_configs = {
                    # tot: 192 * 2 + 79 + 7^2 =  512
                    4: partial_refiner_coarse(
                        feat_dim=256,
                        proj_dim=192,
                        displacement_emb_dim=79,
                        local_corr_radius=3,
                        patch_size=4,
                        block_type="roma",
                    ),
                    # tot: 48 * 2 + 23 + 3^2 = 128
                    2: partial_refiner_coarse(
                        feat_dim=128,
                        proj_dim=48,
                        displacement_emb_dim=23,
                        local_corr_radius=1,
                        patch_size=2,
                        block_type="roma",
                    ),
                    # tot: 12 * 2 + 8 = 32
                    1: partial_refiner_coarse(
                        feat_dim=64,
                        proj_dim=12,
                        displacement_emb_dim=8,
                        local_corr_radius=None,
                        patch_size=1,
                        block_type="roma",
                    ),
                }
        # assert set(cfg.patch_sizes) == set(refiner_configs.keys())
        patch_sizes = list(refiner_configs.keys())
        patch_sizes.sort(reverse=True)
        refiners = nn.ModuleDict()
        # explicit for-loop for order
        for patch_size in patch_sizes:
            refiners[str(patch_size)] = ConvRefiner(refiner_configs[patch_size])
        return refiners
