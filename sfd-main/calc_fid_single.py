import os
import subprocess
import tempfile
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID for one model folder")
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Path to the experiment folder (containing network-snapshot-999999.pkl)"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10",
        help="Dataset name for image generation"
    )
    parser.add_argument(
        "--seeds", type=str, default="0-49999",
        help="Seeds range for image generation"
    )
    parser.add_argument(
        "--batch", type=int, default=256,
        help="Batch size for generation and FID calculation"
    )
    parser.add_argument(
        "--ref_url", type=str,
        default="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        help="Reference statistics URL/path"
    )
    parser.add_argument(
        "--num_images", type=int, default=50000,
        help="Number of images to generate for FID calculation"
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Temporary directory for generated images (default: system temp)"
    )
    return parser.parse_args()


def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    args = parse_args()
    folder = args.folder.rstrip("/")
    folder_name = os.path.basename(folder)

    model_path = os.path.join(folder, "network-snapshot-999999.pkl")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Temp dir
    if args.temp_dir:
        temp_base = args.temp_dir
        os.makedirs(temp_base, exist_ok=True)
    else:
        temp_base = tempfile.gettempdir()
    temp_img_dir = os.path.join(temp_base, f"temp_images_{folder_name}")
    if os.path.exists(temp_img_dir):
        shutil.rmtree(temp_img_dir)
    os.makedirs(temp_img_dir, exist_ok=True)

    # Step 1: Generate images
    cmd_gen = [
        "python", "sample.py",
        f"--model_path={model_path}",
        f"--dataset_name={args.dataset_name}",
        f"--seeds={args.seeds}",
        f"--batch={args.batch}",
        f"--outdir={temp_img_dir}",
        "--subdirs"
    ]
    if not run_cmd(cmd_gen):
        print("Image generation failed")
        return

    # Step 2: Calculate FID
    cmd_fid = [
        "python", "fid.py", "calc",
        f"--images={temp_img_dir}",
        f"--ref={args.ref_url}",
        f"--batch={args.batch}",
        f"--desc={folder_name}"
    ]
    if args.num_images:
        cmd_fid.append(f"--num={args.num_images}")

    if run_cmd(cmd_fid):
        print(f"FID calculation completed for {folder_name}")
    else:
        print(f"FID calculation failed for {folder_name}")

    # Cleanup
    shutil.rmtree(temp_img_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
