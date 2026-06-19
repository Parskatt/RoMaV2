from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import torch
from einops import rearrange
from romav2.normalizers import imagenet
from romav2.types import Normalizer, DescriptorName
from romav2.device import device
from torch import nn
from functools import partial
import torchvision.models as models
from torch.nn import functional as F


def wrap_with_normalize(
    forward: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
):
    def wrapped_forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        with (
            torch.autocast(device.type, torch.bfloat16, enabled=enable_amp),
            torch.set_grad_enabled(not frozen),
        ):
            if self.training and frozen:
                self.eval()
            B, C, H, W = img.shape
            assert C == 3, f"Image must have 3 channels, but got shape {img.shape=}"
            img_n = normalizer(img)
            H = H // patch_size
            W = W // patch_size
            raw_outs = forward(img_n)
            maybe_feat_normalizer = (
                F.normalize if normalize_feats else lambda x, dim=-1: x
            )
            return [
                maybe_feat_normalizer(
                    rearrange(x, "B (H W) D -> B H W D", H=H, W=W), dim=-1
                )
                for x in raw_outs
            ]

    return wrapped_forward


def wrap_model(
    model: nn.Module,
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
    func: Any,
):
    if enable_amp and frozen:  # if training we want params in fp32
        model = model.to(torch.bfloat16)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.frozen = frozen
    type(model).forward = wrap_with_normalize(
        func,
        normalizer=normalizer,
        patch_size=patch_size,
        enable_amp=enable_amp,
        frozen=frozen,
        normalize_feats=normalize_feats,
    )
    return model


def _get_layers(layers, model):
    return [b if b > 0 else len(model.blocks) + b for b in layers]


class Descriptor:
    @dataclass(frozen=True)
    class Cfg:
        name: DescriptorName = "dinov3_vitl16"
        enable_amp: bool = True
        frozen: bool = True
        normalize_feats: bool = False
        layer_idx: list[int] = field(
            default_factory=lambda: [11, 17]
        )  # [4, 11, 17, 23] for dinov3 style
        weights_path: str | None = None
        module_path: str | None = None # Path to local directory containing module

    def __new__(cls, cfg: Cfg) -> nn.Module:
        partial_wrap = partial(
            wrap_model,
            enable_amp=cfg.enable_amp,
            frozen=cfg.frozen,
            normalize_feats=cfg.normalize_feats,
        )
        match cfg.name:
            case "dinov3_vitl16":
                normalizer = imagenet
                # TODO: this will break in distributed if not available locally
                dinov3_vitl16: nn.Module = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov3:adc254450203739c8149213a7a69d8d905b4fcfa" if cfg.module_path is None else cfg.module_path,
                    model="dinov3_vitl16",
                    pretrained=cfg.weights_path is not None,
                    source="github" if cfg.module_path is None else "local",
                    weights=cfg.weights_path,
                    skip_validation=True,
                ).to(device)
                layers = _get_layers(cfg.layer_idx, dinov3_vitl16)
                return partial_wrap(
                    dinov3_vitl16,
                    normalizer=normalizer,
                    patch_size=16,
                    func=partial(dinov3_vitl16.get_intermediate_layers, n=layers),
                )
            case "dinov2_vitl14":
                normalizer = imagenet

                dinov2_vit14: nn.Module = torch.hub.load(
                    repo_or_dir="facebookresearch/dinov2" if cfg.module_path is None else cfg.module_path, 
                    model="dinov2_vitl14",
                    source="github" if cfg.module_path is None else "local"
                ).to(device)
                dinov2_vit14.mask_token = None
                layers = _get_layers(cfg.layer_idx, dinov2_vit14)
                return partial_wrap(
                    dinov2_vit14,
                    normalizer=normalizer,
                    patch_size=14,
                    func=partial(dinov2_vit14.get_intermediate_layers, n=layers),
                )
            case "mum_vitl16":
                raise NotImplementedError("MUM is not supported")
            case _:
                raise ValueError(f"Unknown descriptor name: {cfg.name}")


class VGG(nn.Module):
    def forward(self, x):
        x = imagenet(x)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale = scale * 2
                x = layer(x)
            return feats


class VGG19(VGG):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        if patch_size not in [8]:
            raise NotImplementedError(
                f"VGG19 is not supported for patch size {patch_size}"
            )
        last_layer = {8: 28}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[
                :last_layer
            ]
        )


class VGG19BN(VGG):
    def __init__(self, patch_size: int, pretrained: bool) -> None:
        super().__init__()
        last_layer = {1: 7, 2: 14, 4: 27, 8: 40, 16: 52}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None).features[
                :last_layer
            ]
        )


class FineFeatures(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        type: Literal["vgg19", "vgg19bn"] = "vgg19bn"
        patch_size: int = 4
        pretrained: bool = False

    def __new__(cls, cfg: Cfg):
        match cfg.type:
            case "vgg19":
                return VGG19(cfg.patch_size, cfg.pretrained)
            case "vgg19bn":
                return VGG19BN(cfg.patch_size, cfg.pretrained)
            case _:
                raise ValueError(f"Unknown refiner features type: {cfg.type}")
