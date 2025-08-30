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
    parser = argparse.ArgumentParser(description='Calculate FID for models in all subdirectories')
    parser.add_argument('--base_folder', type=str, default='/home/cherish/SADD/sfd-main/exps',
                       help='Base folder containing subdirectories with model files')
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


def find_model_files_in_subdir(subdir_path):
    """Find the two specific model files in a subdirectory"""
    models = {}
    
    # Find best loss model (pattern: network-snapshot-XXXXXX-loss-X.XXXXXX.pkl)
    loss_pattern = os.path.join(subdir_path, "network-snapshot-*-loss-*.pkl")
    loss_files = glob.glob(loss_pattern)
    if loss_files:
        # If multiple loss files, take the first one (should be only one anyway)
        models['best_loss'] = loss_files[0]
    
    # Find last model (exactly: network-snapshot-999999.pkl)
    last_model = os.path.join(subdir_path, "network-snapshot-999999.pkl")
    if os.path.exists(last_model):
        models['last'] = last_model
    
    return models


def find_all_model_files(base_folder):
    """Find all model files in all subdirectories"""
    all_models = []
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_folder) 
               if os.path.isdir(os.path.join(base_folder, d))]
    
    print(f"Found {len(subdirs)} subdirectories")
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(base_folder, subdir)
        print(f"Checking subdirectory: {subdir}")
        
        models = find_model_files_in_subdir(subdir_path)
        
        for model_type, model_path in models.items():
            all_models.append({
                'path': model_path,
                'subdir': subdir,
                'type': model_type,
                'description': f"{subdir}_{model_type}"
            })
            print(f"  Found {model_type} model: {os.path.basename(model_path)}")
    
    return all_models


def generate_images(model_path, dataset_name, seeds, batch_size, output_dir):
    """Generate images using sample.py"""
    cmd = [
        'python', 'sample.py',
        f'--model_path={model_path}',
        f'--dataset_name={dataset_name}',
        f'--seeds={seeds}',
        f'--batch={batch_size}',
        f'--outdir={output_dir}',
        '--subdirs'
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


def main():
    args = parse_args()
    
    base_folder = args.base_folder
    
    if not os.path.exists(base_folder):
        print(f"Base folder does not exist: {base_folder}")
        return
    
    # Find all model files in all subdirectories
    all_models = find_all_model_files(base_folder)
    
    if not all_models:
        print("No model files found in any subdirectory")
        return
    
    print(f"\nFound {len(all_models)} model files total")
    
    # Create temporary directory for images
    if args.temp_dir:
        temp_base = args.temp_dir
        os.makedirs(temp_base, exist_ok=True)
    else:
        temp_base = tempfile.gettempdir()
    
    successful_models = 0
    failed_models = 0
    
    for i, model_info in enumerate(all_models):
        model_path = model_info['path']
        subdir = model_info['subdir']
        model_type = model_info['type']
        description = model_info['description']
        
        print(f"\n{'='*80}")
        print(f"Processing model {i+1}/{len(all_models)}")
        print(f"Subdirectory: {subdir}")
        print(f"Model type: {model_type}")
        print(f"Model file: {os.path.basename(model_path)}")
        print(f"Description: {description}")
        print(f"{'='*80}")
        
        # Create temporary directory for this model's images
        temp_img_dir = os.path.join(temp_base, f"temp_images_{subdir}_{model_type}")
        
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
                print(f"Failed to generate images for {description}")
                failed_models += 1
                continue
            
            # Calculate FID
            print("Step 2: Calculating FID...")
            fid_success = calculate_fid(
                images_path=temp_img_dir,
                ref_path=args.ref_url,
                batch_size=args.batch,
                description=description,
                num_images=args.num_images
            )
            
            if fid_success:
                print(f"Successfully processed {description}")
                successful_models += 1
            else:
                print(f"FID calculation failed for {description}")
                failed_models += 1
            
        except Exception as e:
            print(f"Error processing {description}: {e}")
            failed_models += 1
        
        finally:
            # Clean up temporary images and memory
            print("Step 3: Cleaning up...")
            cleanup_directory(temp_img_dir)
            cleanup_memory()
            print(f"Completed {description}")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total models processed: {len(all_models)}")
    print(f"Successful: {successful_models}")
    print(f"Failed: {failed_models}")
    print(f"\nAll results have been saved to fid.txt automatically")
    print(f"Each model entry has description format: <subfolder_name>_<model_type>")
    print(f"  - <subfolder_name>_best_loss for lowest loss models")
    print(f"  - <subfolder_name>_last for final saved models")


if __name__ == "__main__":
    main()