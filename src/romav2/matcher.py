import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from romav2.geometry import get_normalized_grid
from einops import einsum
from romav2.device import device
from romav2.vit import ViTModel, vit_from_name
from romav2.types import HeadType, MatcherStyle
from romav2.dpt import DPTHead


def normalize(x: torch.Tensor, dim: int):
    return x / x.norm(dim=dim, keepdim=True)


def cosine_similarity(f_A: torch.Tensor, f_B: torch.Tensor) -> torch.Tensor:
    f_A = normalize(f_A, dim=-1)
    f_B = normalize(f_B, dim=-1)
    res = einsum(f_A, f_B, "B H_A W_A D, B H_B W_B D -> B H_A W_A H_B W_B")
    return res


def _compute_match_embeddings(
    *,
    f_A: torch.Tensor,
    f_B: torch.Tensor,
    pos_emb_grid: torch.Tensor,
    temp: float,
    B: int,
    H_A: int,
    W_A: int,
    H_B: int,
    W_B: int,
    return_attention: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    attn_AB_logits = (1 / temp * cosine_similarity(f_A, f_B)).reshape(
        B, H_A * W_A, H_B * W_B
    )
    attn_AB = torch.softmax(attn_AB_logits, dim=2)
    attn_AB = attn_AB.reshape(B, H_A, W_A, H_B, W_B)

    match_emb = einsum(
        attn_AB, pos_emb_grid, "B H_A W_A H_B W_B, B H_B W_B D -> B H_A W_A D"
    )
    if return_attention:
        attn_AB_logits = attn_AB_logits.reshape(B, H_A, W_A, H_B, W_B)
        return attn_AB_logits, attn_AB, match_emb
    return None, None, match_emb


def _compute_head_preds(
    *,
    f_list_A: list[torch.Tensor],
    match_emb_AB: torch.Tensor,
    f_mv_A: torch.Tensor,
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    head: DPTHead,
) -> torch.Tensor:
    head_input = f_list_A
    head_input[-1] = f_list_A[-1] + f_mv_A + match_emb_AB
    warp_and_confidence = head(head_input, img_A=img_A, img_B=img_B)
    B, H_out, W_out, D_warp_and_confidence = warp_and_confidence.shape
    warp = warp_and_confidence[:, :, :, :2]
    confidence = warp_and_confidence[:, :, :, 2:]
    return warp, confidence


class Matcher(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        mv_vit: ViTModel = "vit_base"
        mv_vit_use_rope: bool = True
        mv_vit_position_mode: Literal["same"] = "same"
        mv_vit_attention_mode: Literal["alternating"] = "alternating"
        head: HeadType = "dpt-no-pos"
        # NOTE: 0.2 in RoMa
        temp: float = 0.1
        # NOTE: 8 in RoMa
        scale: float = 1
        dim: int = 1024
        warp_dim: int = 2
        confidence_dim: int = 1
        num_feature_layers: int = 2
        feat_dim: int = 1024
        pos_emb_dim: int = 1024
        enable_amp: bool = True
        style: MatcherStyle = "romav2"
        # ufm uses pos embedings for view B and no attn
        pos_embed_rope_rescale_coords: float | None = None

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        omega = 2 * torch.pi * torch.randn(cfg.dim // 2, 2)
        self.omega = nn.Buffer(omega)
        self.scale = nn.Buffer(torch.tensor(cfg.scale))

        self.temp = nn.Buffer(torch.tensor(cfg.temp))
        self.mv_vit = vit_from_name(
            cfg.mv_vit,
            device=device,
            in_dim=cfg.feat_dim * cfg.num_feature_layers,
            out_dim=cfg.dim,
            multiview=True,
            use_rope=cfg.mv_vit_use_rope,
            mv_position_mode=cfg.mv_vit_position_mode,
            mv_attention_mode=cfg.mv_vit_attention_mode,
            pos_embed_rope_rescale_coords=cfg.pos_embed_rope_rescale_coords,
        )
        self.head = DPTHead(
            dim_in=cfg.dim,
            out_dim=cfg.warp_dim + cfg.confidence_dim,
            pos_embed=False,
            feature_only=False,
            down_ratio=4,
        )

    def forward(
        self,
        f_list_A: list[torch.Tensor],
        f_list_B: list[torch.Tensor],
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        bidirectional: bool,
        return_attention: bool = True,
    ):
        preds = {}
        f_A = torch.cat(f_list_A, dim=-1)
        f_B = torch.cat(f_list_B, dim=-1)
        B, H_A, W_A, D_feat = f_A.shape
        B, H_B, W_B, D_feat = f_B.shape
        assert D_feat == self.cfg.feat_dim * self.cfg.num_feature_layers, (
            "Feature dimension mismatch"
        )
        x = get_normalized_grid(B, H_B, W_B)
        x_emb = nn.functional.linear(
            x.reshape(B, H_B * W_B, 2), self.scale * self.omega
        ).reshape(B, H_B, W_B, -1)
        pos_emb_grid = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

        with torch.autocast(device.type, torch.bfloat16, enabled=self.cfg.enable_amp):
            assert self.mv_vit is not None
            f_mv_AB = self.mv_vit(torch.stack((f_A, f_B), dim=1))[
                "x_norm_patchtokens"
            ].reshape(B, 2, H_A, W_A, self.cfg.dim)
            f_mv_A = f_mv_AB[:, 0]
            f_mv_B = f_mv_AB[:, 1]

        f_mv_A = f_mv_A.float()
        f_mv_B = f_mv_B.float()

        assert H_A == H_B and W_A == W_B, "H_A and W_A must be equal to H_B and W_B"
        attn_AB_logits, attn_AB, match_emb_AB = _compute_match_embeddings(
            f_A=f_mv_A,
            f_B=f_mv_B,
            pos_emb_grid=pos_emb_grid,
            temp=self.temp,
            B=B,
            H_A=H_A,
            W_A=W_A,
            H_B=H_B,
            W_B=W_B,
            return_attention=return_attention,
        )
        warp_AB, confidence_AB = _compute_head_preds(
            f_list_A=f_list_A,
            match_emb_AB=match_emb_AB,
            f_mv_A=f_mv_A,
            img_A=img_A,
            img_B=img_B,
            head=self.head,
        )

        if bidirectional:
            attn_BA_logits, attn_BA, match_emb_BA = _compute_match_embeddings(
                f_A=f_mv_B,
                f_B=f_mv_A,
                pos_emb_grid=pos_emb_grid,
                temp=self.temp,
                B=B,
                H_A=H_B,
                W_A=W_B,
                H_B=H_A,
                W_B=W_A,
                return_attention=return_attention,
            )
            warp_BA, confidence_BA = _compute_head_preds(
                f_list_A=f_list_B,
                match_emb_AB=match_emb_BA,
                f_mv_A=f_mv_B,
                img_A=img_B,
                img_B=img_A,
                head=self.head,
            )
        else:
            match_emb_BA = None
            attn_BA = None
            attn_BA_logits = None
            warp_BA = None
            confidence_BA = None

        preds["attn_AB_logits"] = attn_AB_logits
        preds["attn_AB"] = attn_AB
        preds["warp_AB"] = warp_AB
        preds["confidence_AB"] = confidence_AB

        preds["attn_BA_logits"] = attn_BA_logits
        preds["attn_BA"] = attn_BA
        preds["warp_BA"] = warp_BA
        preds["confidence_BA"] = confidence_BA
        return preds
