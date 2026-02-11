"""
Sample using pre-built dataset triplets (same as training visualization).
This ensures we use the same high-quality reference pairs used during training.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from css.data.dataset import MegaScenesDataset
from css.models.pose_conditioned_sd import PoseConditionedSD


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] CHW tensor to uint8 HWC numpy."""
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to UNet checkpoint (.pt file)")
    parser.add_argument("--scene", default="MegaScenes/Mysore_Palace", help="Scene directory")
    parser.add_argument("--triplet-idx", type=int, default=0, help="Which dataset triplet to use")
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument("--num-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG scale")
    parser.add_argument("--output", default="sample_ds.png", help="Output path")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = PoseConditionedSD()
    state = torch.load(args.checkpoint, map_location=model.device)
    if isinstance(state, dict) and "unet" in state:
        model.unet.load_state_dict(state["unet"])
        model.ref_encoder.load_state_dict(state["ref_encoder"])
    else:
        model.unet.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset (this will use the same triplet building as training)
    print(f"Loading dataset from {args.scene}...")
    dataset = MegaScenesDataset(
        [args.scene],
        H=512, W=512,
        max_pair_distance=2.0,
        max_triplets_per_scene=10000,
    )
    print(f"Dataset has {len(dataset)} triplets")

    if args.triplet_idx >= len(dataset):
        print(f"Error: triplet_idx {args.triplet_idx} out of range (max: {len(dataset)-1})")
        return

    # Get triplet from dataset
    sample = dataset[args.triplet_idx]
    print(f"\nUsing triplet {args.triplet_idx}")

    # Generate
    print(f"Generating with {args.num_steps} steps, CFG={args.cfg_scale}...")
    with torch.inference_mode():
        generated = model.sample(
            sample["ref1_img"].unsqueeze(0),
            sample["ref2_img"].unsqueeze(0),
            sample["plucker_ref1"].unsqueeze(0),
            sample["plucker_ref2"].unsqueeze(0),
            sample["plucker_target"].unsqueeze(0),
            prompt=args.prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )

    # Create comparison grid
    grid = np.concatenate([
        _to_uint8(sample["ref1_img"]),
        _to_uint8(sample["ref2_img"]),
        _to_uint8(sample["target_img"]),
        _to_uint8(generated[0]),
    ], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"\nSaved: {output_path}")
    print("  [Ref1 | Ref2 | Ground Truth | Generated]")


if __name__ == "__main__":
    main()
