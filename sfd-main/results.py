import time
import subprocess
from pathlib import Path


PROJECT_ROOT = Path("/teamspace/studios/this_studio/SADD/sfd-main")
EXPS_DIR = PROJECT_ROOT / "exps"

DATASET_NAME = "cifar10"
SEEDS = "0-49999"
BATCH = 1024
NUM_IMAGES = 50000
REF_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"

FID_MULTI_SCRIPT = "calc_fid_for_mult_models.py"


def run_cmd(cmd, cwd=None):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)


def list_exp_folders():
    if not EXPS_DIR.exists():
        return []
    return [p for p in EXPS_DIR.iterdir() if p.is_dir()]


def newest_folder():
    folders = list_exp_folders()
    if not folders:
        return None
    return max(folders, key=lambda p: p.stat().st_mtime)


def cmd_has_true_flag(cmd, flag_name):
    target = f"--{flag_name}=True"
    return any(x.strip() == target for x in cmd)


def main():
    train_cmds = [
              [
            "python", "train.py",
            "--dataset_name=cifar10",
            "--total_kimg=200",
            "--batch=128",
            "--lr=5e-5",
            "--num_steps=4",
            "--M=3",
            "--afs=True",
            "--sampler_tea=dpmpp",
            "--max_order=3",
            "--predict_x0=True",
            "--lower_order_final=True",
            "--schedule_type=polynomial",
            "--schedule_rho=7",
            "--use_step_condition=False",
            "--is_second_stage=False",
            "--weight_ls=0.01681693384695571,0.05754291828028599,0.19689602601434278,0.6737239997353861",
            "--seed=1913901889"
        ],
        
              [
            "python", "train.py",
            "--dataset_name=cifar10",
            "--total_kimg=200",
            "--batch=128",
            "--lr=5e-5",
            "--num_steps=5",
            "--M=3",
            "--afs=True",
            "--sampler_tea=dpmpp",
            "--max_order=3",
            "--predict_x0=True",
            "--lower_order_final=True",
            "--schedule_type=polynomial",
            "--schedule_rho=7",
            "--use_step_condition=False",
            "--is_second_stage=False",
            "--weight_ls=0.01681693384695571,0.05754291828028599,0.19689602601434278,0.6737239997353861",
            "--seed=1913901889"
        ],
        
              [
            "python", "train.py",
            "--dataset_name=cifar10",
            "--total_kimg=200",
            "--batch=128",
            "--lr=5e-5",
            "--num_steps=6",
            "--M=3",
            "--afs=True",
            "--sampler_tea=dpmpp",
            "--max_order=3",
            "--predict_x0=True",
            "--lower_order_final=True",
            "--schedule_type=polynomial",
            "--schedule_rho=7",
            "--use_step_condition=False",
            "--is_second_stage=False",
            "--weight_ls=0.01681693384695571,0.05754291828028599,0.19689602601434278,0.6737239997353861",
            "--seed=1913901889"
        ],
        
              [
            "python", "train.py",
            "--dataset_name=cifar10",
            "--total_kimg=200",
            "--batch=128",
            "--lr=5e-5",
            "--num_steps=7",
            "--M=3",
            "--afs=True",
            "--sampler_tea=dpmpp",
            "--max_order=3",
            "--predict_x0=True",
            "--lower_order_final=True",
            "--schedule_type=polynomial",
            "--schedule_rho=7",
            "--use_step_condition=False",
            "--is_second_stage=False",
            "--weight_ls=0.01681693384695571,0.05754291828028599,0.19689602601434278,0.6737239997353861",
            "--seed=1913901889"
        ],
          [
            "python", "train.py",
            "--dataset_name=cifar10",
            "--total_kimg=800",
            "--batch=128",
            "--lr=5e-5",
            "--num_steps=4",
            "--M=3",
            "--afs=True",
            "--sampler_tea=dpmpp",
            "--max_order=3",
            "--predict_x0=True",
            "--lower_order_final=True",
            "--schedule_type=polynomial",
            "--schedule_rho=7",
            "--use_step_condition=True",
            "--is_second_stage=False",
            "--weight_ls=0.01681693384695571,0.05754291828028599,0.19689602601434278,0.6737239997353861",
            "--seed=1913901889"
        ],
        
    ]

    for i, train_cmd in enumerate(train_cmds):
        print(f"\n\n########### TRAIN JOB {i+1}/{len(train_cmds)} ###########\n")

        # 1) TRAIN
        run_cmd(train_cmd, cwd=str(PROJECT_ROOT))

        # 2) wait a bit so folder timestamp is correct
        time.sleep(3)

        # 3) newest folder = this training run folder
        exp_folder = newest_folder()
        if exp_folder is None:
            raise RuntimeError("No experiment folders found in exps/")

        print(f"\n✅ Using newest experiment folder:\n{exp_folder}\n")

        # 4) Decide num_steps usage based on training flag
        use_step_condition = cmd_has_true_flag(train_cmd, "use_step_condition")

        if use_step_condition:
            fid_cmd = [
                "python", FID_MULTI_SCRIPT,
                f"--base_folder={str(exp_folder)}",
                f"--dataset_name={DATASET_NAME}",
                f"--seeds={SEEDS}",
                f"--batch={BATCH}",
                f"--ref_url={REF_URL}",
                f"--num_images={NUM_IMAGES}",
                "--num_steps_list=4,5,6,7"
            ]
        else:
            fid_cmd = [
                "python", FID_MULTI_SCRIPT,
                f"--base_folder={str(exp_folder)}",
                f"--dataset_name={DATASET_NAME}",
                f"--seeds={SEEDS}",
                f"--batch={BATCH}",
                f"--ref_url={REF_URL}",
                f"--num_images={NUM_IMAGES}",
                "--num_steps_list="
            ]

        # 5) RUN FID
        run_cmd(fid_cmd, cwd=str(PROJECT_ROOT))

    print("\n\n✅ ALL TRAINING + FID DONE\n")


if __name__ == "__main__":
    main()
