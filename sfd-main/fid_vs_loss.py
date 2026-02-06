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
        "--weight_ls=0,0,0,1.0",
        "--seed=191390188"
    ]
    # [
    #     "python", "train.py",
    #     "--dataset_name=imagenet64",
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
    #     # "--weight_ls=0.0207275812279243,0.0536014547774357,0.6860375754323215,0.6869277310019198",  9.636435646920336
    #     "--weight_ls=0.0103020854245742,0.0454214912363379,0.2002615762835205,0.8829454481555168",
    #     "--seed=191390188"
    # ]
]

# # # Run training sequentially
# for cmd in train_cmds:
#     print(f"\nRunning training: {' '.join(cmd)}\n")
#     subprocess.run(cmd, check=True)

# After training, run FID calculations
base_folders = [
    # "/home/cherish/SADD/sfd-main/exps/00000-cifar10-4-3-dpmpp-3-poly7.0",
    "/home/cherish/SADD/sfd-main/exps/00075-cifar10-4-2-dpmpp-3-poly7.0-afs",
    # "/teamspace/studios/this_studio/sfd-main/exps/00077-imagenet64-4-2-dpmpp-3-poly7.0-afs",
    # "/home/cherish/SADD/sfd-main/exps/00001-cifar10-4-2-dpmpp-3-poly7.0-afs",
]

for folder in base_folders:
    print(f"\nRunning FID calculation for: {folder}\n")
    subprocess.run(
        # ["python", "calc_fid_single.py", f"--folder={folder}",f"--dataset=imagenet64","--ref_url=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"],
        ["python", "calc_fid_single.py", f"--folder={folder}"],#,f"--dataset=imagenet64","--ref_url=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"],
        check=True
    )



# w1                 w2                 w3                 w4
# 0.0168169338469557 0.0575429182802859 0.1968960260143427 0.6737239997353861  DONE 9.84473411152397
# 0.0150565765163528 0.0573856337276224 0.2187157854074077 0.8335987890877504  DONE 9.723708410188832
# 0.0139210542980094 0.0557095339234981 0.2229394486606637 0.8921632307563885  DONE 9.571101753360923
# 0.0159631598274761 0.0573198447019583 0.2058216939607056 0.7390559050730826  DONE 9.717421789876106
# 0.0148802078120299 0.0533830113237393 0.1915125066792815 0.6870545386088324  DONE 9.639184201278692
# 0.0103020854245742 0.0454214912363379 0.2002615762835205 0.8829454481555168
# 0.0124817892649062 0.0467280578272939 0.1749357677789091 0.6549068005672904
# 0.0139055652373106 0.0564629816899471 0.2292656391101063 0.9309237965008847
# 0.0145401271517582 0.0519263630753616 0.1854417883758488 0.6622581447910041
# 0.0073457756922118 0.0337421581533759 0.1549915603949803 0.7119397545490830
# 0.0086559788539043 0.0353114846971643 0.1440508315192628 0.5876456976915903
# 0.0092997908340821 0.0374601086160265 0.1508915375152103 0.6078000554425295
# # run_dual_train_and_fid.py
# import os
# import time
# import subprocess

# EXPS_DIR = "/teamspace/studios/this_studio/sfd-main/exps"
# IMAGENET64_REF = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz"

# # Expected output run dirs (as you specified)
# RUN_DIR_0 = "/teamspace/studios/this_studio/sfd-main/exps/00072-imagenet64-4-2-dpmpp-3-poly7.0-afs"
# RUN_DIR_1 = "/teamspace/studios/this_studio/sfd-main/exps/00073-imagenet64-4-2-dpmpp-3-poly7.0-afs"

# # Two weight sets (comma-separated; train.py splits on commas)
# WTS_0 = "0.0168169338469557,0.0575429182802859,0.1968960260143427,0.6737239997353861"
# WTS_1 = "0.0150565765163528,0.0573856337276224,0.2187157854074077,0.8335987890877504"

# def start(cmd, gpu_id):
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     # keep CPU contention low when running two jobs
#     env.setdefault("OMP_NUM_THREADS", "4")
#     env.setdefault("MKL_NUM_THREADS", "4")
#     return subprocess.Popen(cmd, env=env)

# def main():
#     # Common train args (desc is auto-constructed by your script)
#     base_train = [
#         "torchrun", "--standalone", "--nproc_per_node=1", "train.py",
#         "--dataset_name=imagenet64",
#         "--total_kimg=200",
#         "--batch=128",
#         "--lr=5e-5",
#         "--num_steps=4",
#         "--M=3",
#         "--afs=True",
#         "--sampler_tea=dpmpp",
#         "--max_order=3",
#         "--predict_x0=True",
#         "--lower_order_final=True",
#         "--schedule_type=polynomial",
#         "--schedule_rho=7",
#         "--use_step_condition=False",
#         "--is_second_stage=False",
#     ]

#     # Training jobs (GPU0 then GPU1, with a 10s gap)
#     train0 = base_train + ["--weight_ls=" + WTS_0, "--seed=191390188", "--outdir", EXPS_DIR]
#     p0 = start(train0, gpu_id=0)

#     time.sleep(10)  # wait 10 seconds before starting the second training

#     train1 = base_train + ["--weight_ls=" + WTS_1, "--seed=191390189", "--outdir", EXPS_DIR]
#     p1 = start(train1, gpu_id=1)

#     # Wait for both trainings to complete
#     rc0 = p0.wait()
#     rc1 = p1.wait()
#     print("Train exit codes:", rc0, rc1)

#     # Kick off FID for both folders in parallel, one per GPU
#     fid0 = [
#         "torchrun", "--standalone", "--nproc_per_node=1", "fid.py", "calc",
#         "--images", RUN_DIR_0,
#         "--ref", IMAGENET64_REF,
#         "--batch", "250"
#     ]
#     fid1 = [
#         "torchrun", "--standalone", "--nproc_per_node=1", "fid.py", "calc",
#         "--images", RUN_DIR_1,
#         "--ref", IMAGENET64_REF,
#         "--batch", "250"
#     ]

#     fp0 = start(fid0, gpu_id=0)
#     fp1 = start(fid1, gpu_id=1)

#     frc0 = fp0.wait()
#     frc1 = fp1.wait()
#     print("FID exit codes:", frc0, frc1)

# if __name__ == "__main__":
#     main()


