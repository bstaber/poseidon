"""Script to run inference using the specified model and dataset.

Usage:
    uv run python experiments/run_inference.py --data_path <DATA_PATH>

All other parameters have default values and can be changed as needed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

def main(
    mode: str, model_path: str, data_path: str, dataset: str, file: str, ckpt_dir: str
):
    cmd = [
        sys.executable,
        root_dir / "src" / "poseidon" / "scOT" / "inference.py",
        "--mode",
        mode,
        "--model_path",
        model_path,
        "--data_path",
        data_path,
        "--dataset",
        dataset,
        "--file",
        file,
        "--ckpt_dir",
        ckpt_dir,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference script with specified parameters."
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode of operation (e.g., eval).",
        default="eval",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model.",
        default=Path.home() / ".cache/huggingface/hub/models--camlab-ethz--Poseidon-L/snapshots/7ffaff436651757ce02f163510e5cd7333782d9f/",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use.",
        default="fluids.incompressible.Gaussians",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing data splits.",
        default="eval.csv",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Directory for checkpoints.",
        default="ckpt_dir",
    )

    args = parser.parse_args()
    main(
        args.mode,
        args.model_path,
        args.data_path,
        args.dataset,
        args.file,
        args.ckpt_dir,
    )
