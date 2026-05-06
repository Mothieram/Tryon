"""Main entry point. Dispatches to GMM / Refinement / Joint training.

Usage:
  python train.py --config configs/config.yaml --stage gmm
  python train.py --config configs/config.yaml --stage refinement \
                  --gmm_checkpoint runs/gmm/best_lpips.pth
  python train.py --config configs/config.yaml --stage joint \
                  --gmm_checkpoint <path> --refine_checkpoint <path>

Stages 'refinement' and 'joint' are wired up in Steps 6 & 7 of the build plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None,
                        choices=["gmm", "refinement", "joint"],
                        help="Override training.stage from the config file.")
    parser.add_argument("--gmm_checkpoint", type=str, default=None)
    parser.add_argument("--refine_checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    stage = args.stage or cfg["training"]["stage"]

    print(f"========== TPS Try-On Training ==========")
    print(f"  config = {args.config}")
    print(f"  stage  = {stage}")
    print(f"==========================================")

    if stage == "gmm":
        from training.train_gmm import train_gmm
        train_gmm(args.config, resume_from=args.resume, smoke=args.smoke)
    elif stage == "refinement":
        try:
            from training.train_refinement import train_refinement
        except ImportError:
            raise SystemExit("Refinement training is part of Step 6 (not yet built).")
        train_refinement(
            args.config,
            gmm_checkpoint=args.gmm_checkpoint,
            resume_from=args.resume,
            smoke=args.smoke,
        )
    elif stage == "joint":
        try:
            from training.train_joint import train_joint
        except ImportError:
            raise SystemExit("Joint training is part of Step 7 (not yet built).")
        train_joint(
            args.config,
            gmm_checkpoint=args.gmm_checkpoint,
            refine_checkpoint=args.refine_checkpoint,
            resume_from=args.resume,
            smoke=args.smoke,
        )
    else:
        raise SystemExit(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
