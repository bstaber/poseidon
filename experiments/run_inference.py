"""Script to run inference using the specified model and dataset.

Usage:
    uv run python experiments/run_inference.py --data_path <DATA_PATH>

All other parameters have default values and can be changed as needed.
"""

import argparse
import os
import subprocess
import sys


def main(
    mode: str, model_path: str, data_path: str, dataset: str, file: str, ckpt_dir: str
):
    cmd = [
        sys.executable,
        os.path.join("src", "poseidon", "scOT", "inference.py"),
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
        required=True,
        help="Mode of operation (e.g., eval).",
        default="eval",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model.",
        default="~/.cache/huggingface/hub/models--camlab-ethz--Poseidon-L/snapshots/7ffaff436651757ce02f163510e5cd7333782d9f/",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use.",
        default="fluids.incompressible.Gaussians",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="File containing data splits.",
        default="eval.csv",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
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
