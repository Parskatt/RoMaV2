import torch
from romav2 import RoMaV2
from romav2.device import device


def test_smoke():
    model = RoMaV2(RoMaV2.Cfg(compile=False))
    model.apply_setting("turbo")

if __name__ == "__main__":
    test_smoke()
