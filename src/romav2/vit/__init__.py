# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.init
from einops import rearrange
from torch import Tensor, nn

from .block import SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
from .layer_scale import LayerScale
from .rms_norm import RMSNorm
from .rope import RopePositionEmbedding
from .utils import named_apply

logger = logging.getLogger(__name__)

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


ViTOutput = Literal[
    "x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens", "x_prenorm", "masks"
]


class MatchTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        use_rope: bool,
        dim: int,
        depth: int,
        num_heads: int,
        ffn_ratio: float,
        multiview: bool,
        mv_position_mode: Literal["same", None],
        mv_attention_mode: Literal["alternating", None],
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        self.projector = nn.Linear(in_dim, dim)
        self.output_projector = nn.Linear(dim, out_dim)
        self.multiview = multiview
        self.mv_position_mode = mv_position_mode
        self.mv_attention_mode = mv_attention_mode
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = (
            dim  # num_features for consistency with other models
        )
        self.n_blocks = depth
        self.num_heads = num_heads

        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, dim, device=device)
            )
        logger.debug(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(dim)
        else:
            self.local_cls_norm = None
        # self.mask_token = nn.Parameter(torch.empty(1, dim, device=device))
        if use_rope:
            self.rope_embed = RopePositionEmbedding(
                dim,
                num_heads=num_heads,
                device=device,
                base=pos_embed_rope_base,
                min_period=pos_embed_rope_min_period,
                max_period=pos_embed_rope_max_period,
                normalize_coords=pos_embed_rope_normalize_coords,
                shift_coords=pos_embed_rope_shift_coords,
                jitter_coords=pos_embed_rope_jitter_coords,
                rescale_coords=pos_embed_rope_rescale_coords,
                dtype=dtype_dict[pos_embed_rope_dtype],
            )
        else:
            self.rope_embed = None

    def init_weights(self):
        if self.rope_embed is not None:
            self.rope_embed._init_weights()
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        named_apply(init_weights_vit, self)

    def prepare_tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int]]:
        if self.multiview:
            B, V, H, W, _ = x.shape
        else:
            B, H, W, _ = x.shape
        x = x.flatten(1, -2)
        x = self.projector(x)
        return x, (H, W)

    def forward_features(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> List[Dict[ViTOutput, Tensor]]:
        assert masks is None
        t2_x, hw_tuple = self.prepare_tokens(x)
        x = t2_x
        H, W = hw_tuple
        for idx, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos: tuple[Tensor, Tensor] = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            if self.multiview and self.mv_attention_mode == "alternating":
                # every kth block run only self attention
                if idx % 2 == 1:
                    x = rearrange(x, "B (V H W) D -> (B V) (H W) D", V=2, H=H, W=W)
                    x = blk(x, rope_sincos)
                    x = rearrange(x, "(B V) (H W) D -> B (V H W) D", V=2, H=H, W=W)
                else:
                    x = blk(x, None)
            else:
                x = blk(x, rope_sincos)
        x_norm = self.norm(x)
        output = {
            "x_norm_patchtokens": self.output_projector(x_norm),
            "x_prenorm": x,
        }
        return output

    def forward(self, *args, **kwargs) -> List[Dict[ViTOutput, Tensor]]:
        ret = self.forward_features(*args, **kwargs)
        return ret


ViTModel = Literal[
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_so400m",
    "vit_huge2",
    "vit_giant2",
    "vit_7b",
]


def vit_from_name(
    name: ViTModel,
    in_dim: int,
    out_dim: int,
    use_rope: bool,
    mv_position_mode: Literal["same", None],
    mv_attention_mode: Literal["alternating", None],
    multiview: bool,
    pos_embed_rope_rescale_coords: float | None,
    device: torch.device | None = None,
) -> MatchTransformer:
    match name:
        case "vit_tiny":
            vit_cls = partial(
                MatchTransformer, dim=192, depth=12, num_heads=3, ffn_ratio=4
            )
        case "vit_small":
            vit_cls = partial(
                MatchTransformer, dim=384, depth=12, num_heads=6, ffn_ratio=4
            )
        case "vit_base":
            vit_cls = partial(
                MatchTransformer, dim=768, depth=12, num_heads=12, ffn_ratio=4
            )
        case "vit_large":
            vit_cls = partial(
                MatchTransformer, dim=1024, depth=24, num_heads=16, ffn_ratio=4
            )
        case "vit_so400m":
            vit_cls = partial(
                MatchTransformer,
                dim=1152,
                depth=27,
                num_heads=18,
                ffn_ratio=3.777777778,
            )
        case "vit_huge2":
            vit_cls = partial(
                MatchTransformer, dim=1280, depth=32, num_heads=20, ffn_ratio=4
            )
        case "vit_giant2":
            vit_cls = partial(
                MatchTransformer, dim=1536, depth=40, num_heads=24, ffn_ratio=4
            )
        case "vit_7b":
            vit_cls = partial(
                MatchTransformer, dim=4096, depth=40, num_heads=32, ffn_ratio=3
            )
        case _:
            raise TypeError(f"Invalid ViT model: {name}")
    return vit_cls(
        in_dim=in_dim,
        out_dim=out_dim,
        use_rope=use_rope,
        device=device,
        multiview=multiview,
        mv_position_mode=mv_position_mode,
        mv_attention_mode=mv_attention_mode,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
    )
