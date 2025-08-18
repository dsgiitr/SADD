import os
import subprocess
import tempfile
import shutil

# paths and constants
dataset_name = "cifar10"
seeds = "0-49999"
batch = 256
ref_stat = "path/to/fid/stat"   # <-- update
models_dir = "models"           # <-- update
fid_results = {}

# loop through all pkl models
for fname in sorted(os.listdir(models_dir)):
    if not fname.endswith(".pkl"):
        continue
    model_num = os.path.splitext(fname)[0]

    print(f"=== Processing model {model_num} ===")

    # create a temp dir for generated images
    temp_dir = tempfile.mkdtemp()

    # 1. generate samples
    subprocess.run([
        "python", "sample.py",
        f"--dataset_name={dataset_name}",
        f"--model_path={model_num}",
        f"--seeds={seeds}",
        f"--batch={batch}",
        f"--outdir={temp_dir}"
    ], check=True)

    # 2. calculate FID
    result = subprocess.run([
        "python", "fid.py", "calc",
        f"--images={temp_dir}",
        f"--ref={ref_stat}"
    ], capture_output=True, text=True, check=True)

    # 3. parse + store result
    fid_output = result.stdout.strip()
    print(f"FID for model {model_num}: {fid_output}")
    fid_results[model_num] = fid_output

    # 4. cleanup
    shutil.rmtree(temp_dir)

# save all results
with open("fid_results.txt", "w") as f:
    for model, fid in fid_results.items():
        f.write(f"{model}: {fid}\n")

print("Done! Results saved to fid_results.txt")
