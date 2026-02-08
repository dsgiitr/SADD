import os
import glob
import subprocess
import shutil
import tempfile
import gc
import torch
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID for all models in a folder")

    parser.add_argument(
        "--base_folder", type=str, required=True,
        help="Base folder containing model files"
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
        "--batch", type=int, default=512,
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

    # ✅ NEW: list of num_steps to evaluate
    # - if empty string => run once without passing --num_steps
    parser.add_argument(
        "--num_steps_list", type=str, default="",
        help="Comma-separated num_steps values. Example: 4,5,6,7. If empty => no --num_steps used."
    )

    return parser.parse_args()


def find_model_files(base_folder):
    """Find all model files matching the pattern network-snapshot-{num}.pkl"""
    pattern = os.path.join(base_folder, "network-snapshot-*.pkl")
    model_files = glob.glob(pattern)

    def extract_number(filepath):
        match = re.search(r"network-snapshot-(\d+)\.pkl", os.path.basename(filepath))
        return int(match.group(1)) if match else 0

    model_files.sort(key=extract_number)
    return model_files


def extract_model_number(model_path):
    """Extract model number from filepath"""
    match = re.search(r"network-snapshot-(\d+)\.pkl", os.path.basename(model_path))
    return int(match.group(1)) if match else 0


def generate_images(model_path, dataset_name, seeds, batch_size, output_dir, num_steps=None):
    """Generate images using sample.py"""
    cmd = [
        "python", "sample.py",
        f"--model_path={model_path}",
        f"--dataset_name={dataset_name}",
        f"--seeds={seeds}",
        f"--batch={batch_size}",
        f"--outdir={output_dir}",
        "--subdirs"
    ]

    # ✅ NEW: only add num_steps if requested
    if num_steps is not None:
        cmd.append(f"--num_steps={num_steps}")

    print(f"Generating images with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Image generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating images: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def calculate_fid(images_path, ref_path, batch_size, description, num_images=None):
    """Calculate FID using fid.py"""
    cmd = [
        "python", "fid.py", "calc",
        f"--images={images_path}",
        f"--ref={ref_path}",
        f"--batch={batch_size}",
        f"--desc={description}"
    ]

    if num_images:
        cmd.append(f"--num={num_images}")

    print(f"Calculating FID with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("FID calculation completed successfully")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error calculating FID: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def cleanup_memory():
    """Clean up GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def cleanup_directory(directory):
    """Remove directory and all its contents"""
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)


def main():
    args = parse_args()

    base_folder = args.base_folder.rstrip("/")
    base_folder_name = os.path.basename(os.path.abspath(base_folder))

    # Find all model files
    model_files = find_model_files(base_folder)
    if not model_files:
        print(f"No model files found in {base_folder}")
        return

    print(f"Found {len(model_files)} model files")

    # Create temporary directory for images
    if args.temp_dir:
        temp_base = args.temp_dir
        os.makedirs(temp_base, exist_ok=True)
    else:
        temp_base = tempfile.gettempdir()

    # ✅ Parse num_steps_list
    raw_steps = args.num_steps_list.strip()
    if raw_steps == "":
        num_steps_list = [None]  # means run once with no --num_steps
    else:
        num_steps_list = [int(x.strip()) for x in raw_steps.split(",") if x.strip()]

    successful_runs = 0
    failed_runs = 0

    for i, model_path in enumerate(model_files):
        model_num = extract_model_number(model_path)

        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(model_files)}: {os.path.basename(model_path)}")
        print(f"Model number: {model_num}")
        print(f"{'='*60}")

        for num_steps in num_steps_list:
            # temp dir name must include num_steps so it doesn't conflict
            step_tag = "nosteps" if num_steps is None else f"steps{num_steps}"
            temp_img_dir = os.path.join(temp_base, f"temp_images_model_{model_num}_{step_tag}")

            try:
                cleanup_directory(temp_img_dir)
                os.makedirs(temp_img_dir, exist_ok=True)

                # -------------------------
                # Step 1: Generate images
                # -------------------------
                print("Step 1: Generating images...")

                success = generate_images(
                    model_path=model_path,
                    dataset_name=args.dataset_name,
                    seeds=args.seeds,
                    batch_size=args.batch,
                    output_dir=temp_img_dir,
                    num_steps=num_steps
                )

                if not success:
                    print(f"Failed to generate images for model {model_num} (num_steps={num_steps})")
                    failed_runs += 1
                    continue

                # -------------------------
                # Step 2: Calculate FID
                # -------------------------
                print("Step 2: Calculating FID...")

                if num_steps is None:
                    description = f"{base_folder_name}_network_snapshot_{model_num}"
                else:
                    description = f"{base_folder_name}_network_snapshot_{model_num}_num_steps_{num_steps}"

                fid_success = calculate_fid(
                    images_path=temp_img_dir,
                    ref_path=args.ref_url,
                    batch_size=args.batch,
                    description=description,
                    num_images=args.num_images
                )

                if fid_success:
                    print(f"✅ Successfully processed model {model_num} (num_steps={num_steps})")
                    successful_runs += 1
                else:
                    print(f"❌ FID calculation failed for model {model_num} (num_steps={num_steps})")
                    failed_runs += 1

            except Exception as e:
                print(f"Error processing model {model_num} (num_steps={num_steps}): {e}")
                failed_runs += 1

            finally:
                print("Step 3: Cleaning up...")
                cleanup_directory(temp_img_dir)
                cleanup_memory()
                print(f"Completed model {model_num} (num_steps={num_steps})")

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total snapshots: {len(model_files)}")
    print(f"Total FID runs: {len(model_files) * len(num_steps_list)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print("\nAll results have been saved to fid.txt automatically (from fid.py).")


if __name__ == "__main__":
    main()
