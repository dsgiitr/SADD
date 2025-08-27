# import subprocess

# # List of base folders
# base_folders = [
#     "/home/cherish/SADD/sfd-main/exps/00000-cifar10-4-3-dpmpp-3-poly7.0",
#     "/home/cherish/SADD/sfd-main/exps/00001-cifar10-4-3-dpmpp-3-poly7.0",
# ]

# # Run the command for each base folder sequentially
# for folder in base_folders:
#     print(f"\nRunning FID calculation for: {folder}\n")
#     subprocess.run(
#         ["python", "cal_fid_for_mult_models.py", f"--base_folder={folder}"],
#         check=True
#     )


import subprocess

# Training commands
train_cmds = [
    # [
    #     "python", "train.py",
    #     "--dataset_name=cifar10",
    #     "--total_kimg=200",
    #     "--batch=128",
    #     "--lr=5e-5",
    #     "--num_steps=4",
    #     "--M=3",
    #     "--afs=True",
    #     "--sampler_tea=dpmpp",
    #     "--max_order=3",
    #     "--predict_x0=True",
    #     "--lower_order_final=True",
    #     "--schedule_type=polynomial",
    #     "--schedule_rho=7",
    #     "--use_step_condition=False",
    #     "--is_second_stage=False",
    #     "--weight_ls=0,0,0,1.0",
    #     "--seed=191390188"
    # ]
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
        "--weight_ls=0.0042,0.0637,0.328,1.0",
        "--seed=191390188"
    ]
]

# # Run training sequentially
for cmd in train_cmds:
    print(f"\nRunning training: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

# After training, run FID calculations
base_folders = [
    # "/home/cherish/SADD/sfd-main/exps/00000-cifar10-4-3-dpmpp-3-poly7.0",
    # "/home/cherish/SADD/sfd-main/exps/00000-cifar10-4-3-dpmpp-3-poly7.0",
    "/home/cherish/SADD/sfd-main/exps/00000-cifar10-4-2-dpmpp-3-poly7.0-afs",
    "/home/cherish/SADD/sfd-main/exps/00001-cifar10-4-2-dpmpp-3-poly7.0-afs",
]

for folder in base_folders:
    print(f"\nRunning FID calculation for: {folder}\n")
    subprocess.run(
        ["python", "cal_fid_for_mult_models.py", f"--base_folder={folder}"],#,"--max_workers=4"],
        check=True
    )
