import torch

from romav2.refiner import ConvRefiner


def test_refiner_uses_input_device_on_cpu():
    refiner = ConvRefiner(
        ConvRefiner.Cfg(
            feat_dim=4,
            proj_dim=2,
            displacement_emb_dim=2,
            local_corr_radius=None,
            patch_size=4,
            hidden_blocks=1,
            enable_amp=False,
        )
    ).cpu()
    features_a = torch.rand(1, 4, 4, 4)
    features_b = torch.rand(1, 4, 4, 4)
    previous_warp = torch.zeros(1, 4, 4, 2)
    scale_factor = torch.ones(2)

    result = refiner(
        f_A=features_a,
        f_B=features_b,
        prev_warp=previous_warp,
        prev_confidence=None,
        scale_factor=scale_factor,
    )

    assert result["warp"].device.type == "cpu"
    assert result["confidence"].device.type == "cpu"
