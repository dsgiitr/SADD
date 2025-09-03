# import subprocess
# import re
# import csv
# import os
# import optuna
# from scipy.stats import uniform  # you can remove if unused

# # ---------- log parsing ----------

# def find_min_s2_loss(log_path: str) -> float:
#     """
#     Parse the log file to extract all Step 2 Loss_ls-3-mean values
#     and return their minimum.
#     """
#     step2_ls3_losses = []
#     step2_ls3_pattern = re.compile(r"Step:\s*2\s*\|\s*Loss_ls-3-mean:\s*([0-9]+\.[0-9]+)")

#     with open(log_path, 'r') as f:
#         for line in f:
#             match = step2_ls3_pattern.search(line)
#             if match:
#                 loss_value = float(match.group(1))
#                 step2_ls3_losses.append(loss_value)

#     if not step2_ls3_losses:
#         raise ValueError(f"No Step 2 Loss_ls-3-mean found in {log_path}")

#     return min(step2_ls3_losses)

# def find_last_tick(log_path: str) -> int:
#     """
#     Parse the log file to find the tick value from the last 'tick' line.
#     """
#     tick_pattern = re.compile(r"tick\s+(\d+)\b")
#     last_tick = None

#     with open(log_path, 'r') as f:
#         for line in f:
#             match = tick_pattern.search(line)
#             if match:
#                 last_tick = int(match.group(1))

#     return last_tick

# # ---------- Optuna objective factory ----------

# def make_objective(base_exp_id: int, log_dir: str, train_script="train.py", extra_cmd=None):
#     """
#     Returns an Optuna objective function which runs the external training script (subprocess),
#     then parses the produced log and returns the objective (min step2 loss).
#     """

#     # ensure log_dir exists
#     os.makedirs(log_dir, exist_ok=True)

#     def objective(trial: optuna.trial.Trial):
#         # Suggest hyperparameters (same ranges as original)
#         a = trial.suggest_float("a", 0.0, 1.0)
#         b = trial.suggest_float("b", 0.0, 1.0)
#         c = trial.suggest_float("c", 0.0, 1.0)
#         d = trial.suggest_float("d", 0.0, 1.0)
#         # lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)

#         # compute weights exactly like your original script
#         w1 = a 
#         w2 = a + b
#         w3 = a + b + c
#         w4 = a + b + c + d
#         weights = [w1, w2, w3, w4]

#         # experiment id and folder naming (unique)
#         exp_idx = trial.number + 1 + base_exp_id
#         # exp_folder = f"{exp_idx:05d}-cifar10-4-3-dpmpp-3-poly7.0"
#         exp_folder = f"{exp_idx:05d}-cifar10-4-2-dpmpp-3-poly7.0-afs"
#         exp_path = os.path.join(log_dir, exp_folder)
#         log_path = os.path.join(exp_path, "log.txt")

#         print(f"[Trial {trial.number}] a,b,c = {[a, b, c]}")
#         print(f"[Trial {trial.number}] weights = {weights}")
#         print(f"[Trial {trial.number}] exp_folder = {exp_folder}")

#         # build the command; redirect stdout/stderr to log_path so we can parse it
#         cmd = [
#             "python", train_script,
#             "--dataset_name=cifar10",
#             "--total_kimg=200",
#             "--batch=128",
#             f"--lr=5e-5",
#             "--num_steps=4",
#             "--M=3",
#             "--afs=True",
#             "--sampler_tea=dpmpp",
#             "--max_order=3",
#             "--predict_x0=True",
#             "--lower_order_final=True",
#             "--schedule_type=polynomial",
#             "--schedule_rho=7",
#             "--use_step_condition=False",
#             "--is_second_stage=False",
#             "--use_repeats=True",
#             "--seed=1913901889",
#             f"--weight_ls={','.join(f'{w}' for w in weights)}"
#         ]

#         # allow optional extra command line args
#         if extra_cmd:
#             cmd.extend(extra_cmd)

#         # # Run the subprocess and capture its output into log.txt
#         # with open(log_path, "w") as log_file:
#         #     try:
#         #         subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
#         #     except subprocess.CalledProcessError as e:
#         #         # if the training fails, mark trial as failed
#         #         print(f"[Trial {trial.number}] training failed (subprocess error): {e}")
#         #         # You can either return a large loss or raise to mark as failed
#         #         raise
#         subprocess.run(cmd, check=True)

#         # after successful run, parse the log for objective
#         try:
#             loss = find_min_s2_loss(log_path)
#         except Exception as e:
#             print(f"[Trial {trial.number}] failed to parse log: {e}")
#             raise

#         # store useful metadata in trial attributes
#         trial.set_user_attr("exp_folder", exp_folder)
#         trial.set_user_attr("weights", weights)
#         trial.set_user_attr("log_path", log_path)

#         # optionally report to optuna (no intermediate reporting here since training is external)
#         trial.report(loss, step=0)  # final report

#         print(f"[Trial {trial.number}] loss = {loss}\n")
#         return loss

#     return objective


# # ---------- main ----------

# if __name__ == "__main__":
#     BASE_EXP_ID = 0
#     LOG_DIR = "/home/cherish/SADD/sfd-main/exps"

#     # Make an Optuna study: use TPE (Bayesian-ish) sampler + a median pruner (works if you report intermediate values)
#     # If you want distributed storage use SQLite or RDB: storage="sqlite:///optuna_study.db"
#     study = optuna.create_study(
#         study_name="diffusion_distill_weights",
#         direction="minimize",
#         sampler=optuna.samplers.TPESampler(),
#         pruner=optuna.pruners.MedianPruner()  # prune when possible (requires intermediate reports)
#     )

#     objective = make_objective(BASE_EXP_ID - 1, LOG_DIR)

#     # config
#     conf = {
#         "num_iteration": 50,
#         "n_jobs": 1,           # change to >1 to parallelize trials (ensure train.py can run concurrently)
#     }

#     try:
#         study.optimize(objective, n_trials=conf["num_iteration"], n_jobs=conf["n_jobs"])
#     except KeyboardInterrupt:
#         print("Interrupted by user. Proceeding to save results...")

#     # write CSV summary from the trials
#     csv_file = "evaluation_summary_optuna.csv"
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["trial_number", "exp_folder", "a", "b", "c","d","w1", "w2", "w3", "w4", "loss", "state"])
#         for t in study.trials:
#             attrs = t.user_attrs
#             weights = attrs.get("weights", [None, None, None, None])
#             # If params exist, get a,b,c
#             a = t.params.get("a")
#             b = t.params.get("b")
#             c = t.params.get("c")
#             d = t.params.get("d")
#             writer.writerow([t.number, attrs.get("exp_folder", ""), a, b, c, d,*weights, t.value, t.state.name])

#     print("Best params:", study.best_trial.params)
#     print("Best loss:", study.best_value)
#     print(f"Saved all runs to {csv_file}")




import subprocess
import re
import csv
import os
import optuna

# ---------- FID parsing ----------

def find_fid_score(fid_file_path: str, exp_folder: str) -> float:
    """
    Parse the fid.txt file to extract FID score for the given experiment folder.
    Looks for lines like: "exp_folder_description FID_SCORE"
    """
    if not os.path.exists(fid_file_path):
        raise FileNotFoundError(f"FID file not found: {fid_file_path}")
    
    with open(fid_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(exp_folder):
                # Extract the last number from the line (FID score)
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        fid_score = float(parts[-1])
                        return fid_score
                    except ValueError:
                        continue
    
    raise ValueError(f"No FID score found for {exp_folder} in {fid_file_path}")

# ---------- Optuna objective factory ----------

def make_objective(base_exp_id: int, log_dir: str, fid_file: str, train_script="train.py"):
    """
    Returns an Optuna objective function that:
    1. Runs training with hyperparameters
    2. Runs FID calculation 
    3. Parses FID score from fid.txt
    4. Returns FID score (to minimize)
    """
    
    os.makedirs(log_dir, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        # Suggest hyperparameters
        a = trial.suggest_float("a", 0.0, 1.0)
        b = trial.suggest_float("b", 0.0, 1.0)
        c = trial.suggest_float("c", 0.0, 1.0)
        d = trial.suggest_float("d", 0.0, 1.0)

        # Compute weights
        w1 = a 
        w2 = a + b
        w3 = a + b + c
        w4 = a + b + c + d
        weights = [w1, w2, w3, w4]

        # Experiment naming
        exp_idx = trial.number + base_exp_id
        
        exp_folder = f"{exp_idx:05d}-cifar10-4-2-dpmpp-3-poly7.0-afs"
        exp_path = os.path.join(log_dir, exp_folder)

        print(f"[Trial {trial.number}] Running experiment: {exp_folder}")
        print(f"[Trial {trial.number}] weights = {weights}")

        # Step 1: Run training
        train_cmd = [
            "python", train_script,
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
            "--use_repeats=True",
            "--seed=1913901889",
            f"--weight_ls={','.join(f'{w}' for w in weights)}"
        ]

        try:
            print(f"[Trial {trial.number}] Starting training...")
            subprocess.run(train_cmd, check=True)
            print(f"[Trial {trial.number}] Training completed")
        except subprocess.CalledProcessError as e:
            print(f"[Trial {trial.number}] Training failed: {e}")
            raise

        # Step 2: Run FID calculation
        fid_cmd = [
            "python", "calc_fid_single.py",  # assuming your FID script
            f"--folder={exp_path}",
        ]

        try:
            print(f"[Trial {trial.number}] Calculating FID...")
            subprocess.run(fid_cmd, check=True)
            print(f"[Trial {trial.number}] FID calculation completed")
        except subprocess.CalledProcessError as e:
            print(f"[Trial {trial.number}] FID calculation failed: {e}")
            raise

        # Step 3: Parse FID score
        try:
            fid_score = find_fid_score(fid_file, exp_folder)
            print(f"[Trial {trial.number}] FID score: {fid_score}")
        except Exception as e:
            print(f"[Trial {trial.number}] Failed to parse FID: {e}")
            raise

        # Store metadata
        trial.set_user_attr("exp_folder", exp_folder)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("fid_score", fid_score)

        return fid_score

    return objective

# ---------- main ----------

if __name__ == "__main__":
    BASE_EXP_ID = 0  # adjust based on your existing experiments
    LOG_DIR = "/home/cherish/SADD/sfd-main/exps"
    FID_FILE = "/home/cherish/SADD/sfd-main/fid.txt"

    # Create Optuna study
    study = optuna.create_study(
        study_name="diffusion_fid_optimization",
        direction="minimize",  # minimize FID
        sampler=optuna.samplers.TPESampler(),
    )

    objective = make_objective(BASE_EXP_ID, LOG_DIR, FID_FILE)

    # Configuration
    N_TRIALS = 50  # adjust as needed

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Interrupted by user. Proceeding to save results...")

    # Save results to CSV
    csv_file = "fid_optimization_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "exp_folder", "a", "b", "c", "d", "w1", "w2", "w3", "w4", "fid_score", "state"])
        
        for t in study.trials:
            attrs = t.user_attrs
            weights = attrs.get("weights", [None, None, None, None])
            a = t.params.get("a")
            b = t.params.get("b") 
            c = t.params.get("c")
            d = t.params.get("d")
            fid_score = attrs.get("fid_score")
            
            writer.writerow([
                t.number, 
                attrs.get("exp_folder", ""), 
                a, b, c, d, 
                *weights, 
                fid_score, 
                t.state.name
            ])

    print(f"\nOptimization completed!")
    print(f"Best params: {study.best_trial.params}")
    print(f"Best FID score: {study.best_value}")
    print(f"Results saved to {csv_file}")