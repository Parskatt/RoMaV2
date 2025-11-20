from pathlib import Path

import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romav2.io import tensor_to_pil
from romav2.device import device
from romav2 import RoMaV2

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument(
        "--save_path", default="demo/roma_v2_warp_toronto.jpg", type=str
    )

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model

    model = RoMaV2()
    model.apply_setting("precise")
    H, W = (model.H_lr, model.W_lr) if (model.H_hr is None or model.W_hr is None) else (model.H_hr, model.W_hr) 
    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    preds = model.match(im1_path, im2_path)
    warp_AB, overlap_AB = preds["warp_AB"][0], preds["overlap_AB"][0]
    warp_BA, overlap_BA = preds["warp_BA"][0], preds["overlap_BA"][0]
    # Sampling not needed, but can be done with model.sample(warp, certainty)

    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
        x2[None], warp_AB[None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1[None], warp_BA[None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    overlap = torch.cat((overlap_AB, overlap_BA), dim=1)[..., 0]
    white_im = torch.ones((H, 2 * W), device=device)
    vis_im = overlap * warp_im + (1 - overlap) * white_im
    if not Path(save_path).exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(vis_im).save(save_path)
