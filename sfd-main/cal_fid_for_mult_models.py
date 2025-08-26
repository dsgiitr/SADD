import os
import glob
import subprocess
import shutil
import tempfile
import gc
import torch
from pathlib import Path
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FID for all models in a folder')
    parser.add_argument('--base_folder', type=str, required=True, 
                       help='Base folder containing model files')
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                       help='Dataset name for image generation')
    parser.add_argument('--seeds', type=str, default='0-49999',
                       help='Seeds range for image generation')
    parser.add_argument('--batch', type=int, default=256,
                       help='Batch size for generation and FID calculation')
    parser.add_argument('--ref_url', type=str, 
                       default='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
                       help='Reference statistics URL/path')
    parser.add_argument('--num_images', type=int, default=50000,
                       help='Number of images to generate for FID calculation')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='Temporary directory for generated images (default: system temp)')
    return parser.parse_args()


def find_model_files(base_folder):
    """Find all model files matching the pattern network-snapshot-{num}.pkl"""
    pattern = os.path.join(base_folder, "network-snapshot-*.pkl")
    
    model_files = glob.glob(pattern)
    
    # Sort by model number
    def extract_number(filepath):
        match = re.search(r'network-snapshot-(\d+)\.pkl', os.path.basename(filepath))
        return int(match.group(1)) if match else 0
    
    model_files.sort(key=extract_number)
    return model_files


def generate_images(model_path, dataset_name, seeds, batch_size, output_dir):
    """Generate images using sample.py"""
    cmd = [
        'python', 'sample.py',
        f'--model_path={model_path}',
        f'--dataset_name={dataset_name}',
        f'--seeds={seeds}',
        f'--batch={batch_size}',
        f'--outdir={output_dir}',
        '--subdirs'  # Enable subdirs flag
    ]
    
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
        'python', 'fid.py', 'calc',
        f'--images={images_path}',
        f'--ref={ref_path}',
        f'--batch={batch_size}',
        f'--desc={description}'
    ]
    
    if num_images:
        cmd.append(f'--num={num_images}')
    
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
        try:
            shutil.rmtree(directory)
            print(f"Cleaned up directory: {directory}")
        except Exception as e:
            print(f"Error cleaning up directory {directory}: {e}")


def extract_model_number(model_path):
    """Extract model number from filepath"""
    match = re.search(r'network-snapshot-(\d+)\.pkl', os.path.basename(model_path))
    return int(match.group(1)) if match else 0


def main():
    args = parse_args()
    
    base_folder = args.base_folder
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
    
    successful_models = 0
    failed_models = 0
    
    for i, model_path in enumerate(model_files):
        model_num = extract_model_number(model_path)
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(model_files)}: {os.path.basename(model_path)}")
        print(f"Model number: {model_num}")
        print(f"{'='*60}")
        
        # Create temporary directory for this model's images
        temp_img_dir = os.path.join(temp_base, f"temp_images_model_{model_num}")
        
        try:
            # Clean up any existing temp directory
            cleanup_directory(temp_img_dir)
            os.makedirs(temp_img_dir, exist_ok=True)
            
            # Generate images
            print("Step 1: Generating images...")
            success = generate_images(
                model_path=model_path,
                dataset_name=args.dataset_name,
                seeds=args.seeds,
                batch_size=args.batch,
                output_dir=temp_img_dir
            )
            
            if not success:
                print(f"Failed to generate images for model {model_num}")
                failed_models += 1
                continue
            
            # Calculate FID with descriptive name
            print("Step 2: Calculating FID...")
            description = f"{base_folder_name}_netwrol_snapshot_{model_num}"
            
            fid_success = calculate_fid(
                images_path=temp_img_dir,
                ref_path=args.ref_url,
                batch_size=args.batch,
                description=description,
                num_images=args.num_images
            )
            
            if fid_success:
                print(f"Successfully processed model {model_num}")
                successful_models += 1
            else:
                print(f"FID calculation failed for model {model_num}")
                failed_models += 1
            
        except Exception as e:
            print(f"Error processing model {model_num}: {e}")
            failed_models += 1
        
        finally:
            # Clean up temporary images and memory
            print("Step 3: Cleaning up...")
            cleanup_directory(temp_img_dir)
            cleanup_memory()
            print(f"Completed model {model_num}")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total models processed: {len(model_files)}")
    print(f"Successful: {successful_models}")
    print(f"Failed: {failed_models}")
    print(f"\nAll results have been saved to fid.txt automatically")
    print(f"Each model entry in fid.txt has description: {base_folder_name}_netwrol_snapshot_<model_num>")


if __name__ == "__main__":
    main()
