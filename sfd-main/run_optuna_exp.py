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
from typing import Optional, Tuple
import math

# ---------- Robust folder detection ----------

def find_latest_experiment_folder(log_dir: str, base_pattern: str = "cifar10") -> Tuple[str, str]:
    """
    Find the most recent experiment folder by checking both:
    1. Highest experiment number
    2. Last modified timestamp
    Returns: (folder_name, full_path)
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Get all folders matching the pattern
    folders = [f for f in os.listdir(log_dir) 
               if os.path.isdir(os.path.join(log_dir, f)) and base_pattern in f]
    
    if not folders:
        raise ValueError(f"No experiment folders found in {log_dir}")
    
    # Method 1: Find folder with highest experiment number
    exp_numbers = []
    for folder in folders:
        match = re.match(r'^(\d+)-', folder)
        if match:
            exp_numbers.append((int(match.group(1)), folder))
    
    if exp_numbers:
        latest_by_number = max(exp_numbers, key=lambda x: x[0])[1]
    else:
        latest_by_number = None
    
    # Method 2: Find last modified folder
    folder_times = [(f, os.path.getmtime(os.path.join(log_dir, f))) for f in folders]
    latest_by_time = max(folder_times, key=lambda x: x[1])[0]
    
    # Use the one with highest number, fallback to last modified
    selected_folder = latest_by_number if latest_by_number else latest_by_time
    
    print(f"  Latest by number: {latest_by_number}")
    print(f"  Latest by modification time: {latest_by_time}")
    print(f"  Selected folder: {selected_folder}")
    
    return selected_folder, os.path.join(log_dir, selected_folder)


# ---------- FID parsing ----------

def find_fid_score(fid_file_path: str, exp_folder: str) -> float:
    """
    Parse the fid.txt file to extract FID score for the given experiment folder.
    Looks for lines like: "exp_folder_description FID_SCORE"
    Falls back to last entry if exact match not found.
    """
    if not os.path.exists(fid_file_path):
        raise FileNotFoundError(f"FID file not found: {fid_file_path}")
    
    with open(fid_file_path, 'r') as f:
        lines = f.readlines()
    
    # Try exact match first
    for line in lines:
        line = line.strip()
        if line.startswith(exp_folder):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    fid_score = float(parts[-1])
                    print(f"  Found FID score by exact match: {fid_score}")
                    return fid_score
                except ValueError:
                    continue
    
    # If no exact match, try to find the last entry in the file
    print(f"  Warning: No exact match for {exp_folder}, using last entry...")
    for line in reversed(lines):
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    fid_score = float(parts[-1])
                    print(f"  Using last FID entry in file: {fid_score}")
                    return fid_score
                except ValueError:
                    continue
    
    raise ValueError(f"No FID score found for {exp_folder} in {fid_file_path}")


# ---------- CSV updater ----------

def append_trial_to_csv(csv_file: str, trial: optuna.trial.FrozenTrial):
    """
    Append a single trial's results to the CSV file.
    Creates the file with headers if it doesn't exist.
    """
    if trial.state == optuna.trial.TrialState.PRUNED:
        print(f"[Trial {trial.number}] Skipped (pruned).")
        return
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "trial_number", "exp_folder", "a", "b", "constraint_value",
                "w1", "w2", "w3", "w4", "fid_score", "state"
            ])
        
        # Extract trial data
        attrs = trial.user_attrs
        a = attrs.get("a")
        b = attrs.get("b")
        constraint_value = attrs.get("constraint_value")
        weights = attrs.get("weights", [])
        fid_score = attrs.get("fid_score")
        
        # Write trial row
        writer.writerow([
            trial.number,
            attrs.get("exp_folder", ""),
            a, b, constraint_value,
            *weights,
            fid_score,
            trial.state.name
        ])


# ---------- Optuna objective factory with reparametrization ----------

def make_objective(base_exp_id: int, log_dir: str, fid_file: str, 
                   csv_file: str, train_script="train.py"):
    os.makedirs(log_dir, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        # REPARAMETRIZATION: Sample a and constraint directly
        # This ensures the search space is consistent across all trials
        
        # Sample a from its full range
        a = trial.suggest_float("a", 0.5, 3.0)
        
        # Sample the constraint value directly from its valid range
        # constraint = 3a + b, so we sample constraint ∈ [ln(1/4), ln(4)]
        constraint = trial.suggest_float("constraint", math.log(1/4), math.log(4))
        
        # Derive b from the constraint
        b = constraint - 3*a
        
        # Verify b is within the acceptable range [-7, -2]
        if b < -7.0 or b > -2.0:
            # This combination violates the original b bounds
            print(f"[Trial {trial.number}] Derived b={b:.4f} out of bounds [-7, -2]")
            raise optuna.TrialPruned(f"Derived b={b:.4f} violates bounds")
        
        # Store derived parameters as user attributes for logging
        trial.set_user_attr("a", a)
        trial.set_user_attr("b", b)
        trial.set_user_attr("constraint_value", constraint)

        # Compute weights
        weights = [math.exp(b + i*a) for i in range(4)]
        
        # Experiment naming
        exp_idx = trial.number + base_exp_id
        exp_folder = f"{exp_idx:05d}-cifar10-4-2-dpmpp-3-poly7.0-afs"
        exp_path = os.path.join(log_dir, exp_folder)

        print(f"\n[Trial {trial.number}] Running experiment: {exp_folder}")
        print(f"[Trial {trial.number}] Parameters: a={a:.4f}, constraint={constraint:.4f}")
        print(f"[Trial {trial.number}] Derived b={b:.4f}")
        print(f"[Trial {trial.number}] Weights = {[f'{w:.6f}' for w in weights]}")

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

        # Step 2: Detect the actual experiment folder that was created
        try:
            print(f"[Trial {trial.number}] Detecting actual experiment folder...")
            actual_folder, actual_path = find_latest_experiment_folder(log_dir)
            
            # Update exp_folder if it's different
            if actual_folder != exp_folder:
                print(f"[Trial {trial.number}] Note: Expected '{exp_folder}' but found '{actual_folder}'")
                exp_folder = actual_folder
                exp_path = actual_path
        except Exception as e:
            print(f"[Trial {trial.number}] Warning: Could not detect actual folder: {e}")
            print(f"[Trial {trial.number}] Continuing with expected folder name...")

        # Step 3: Run FID calculation
        fid_cmd = [
            "python", "calc_fid_single.py",
            f"--folder={exp_path}",
        ]

        try:
            print(f"[Trial {trial.number}] Calculating FID...")
            subprocess.run(fid_cmd, check=True)
            print(f"[Trial {trial.number}] FID calculation completed")
        except subprocess.CalledProcessError as e:
            print(f"[Trial {trial.number}] FID calculation failed: {e}")
            raise

        # Step 4: Parse FID score
        try:
            print(f"[Trial {trial.number}] Parsing FID score...")
            fid_score = find_fid_score(fid_file, exp_folder)
            print(f"[Trial {trial.number}] ✓ FID score: {fid_score}")
        except Exception as e:
            print(f"[Trial {trial.number}] Failed to parse FID: {e}")
            raise

        # Store metadata
        trial.set_user_attr("exp_folder", exp_folder)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("fid_score", fid_score)

        return fid_score

    return objective


# ---------- Callback for CSV updates ----------

class CSVCallback:
    """Callback to update CSV after each trial"""
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
    
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        append_trial_to_csv(self.csv_file, trial)
        print(f"[Trial {trial.number}] Results appended to {self.csv_file}")


# ---------- main ----------

if __name__ == "__main__":
    BASE_EXP_ID = 0
    LOG_DIR = "/home/cherish/SADD/sfd-main/exps"
    FID_FILE = "/home/cherish/SADD/sfd-main/fid.txt"
    CSV_FILE = "fid_optimization_results.csv"

    # Use SQLite for persistent storage
    DB_PATH = "sqlite:///fid_optimization.db"

    # Create or load Optuna study
    study = optuna.create_study(
        study_name="diffusion_fid_optimization",
        storage=DB_PATH,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),  # Added seed for reproducibility
        load_if_exists=True
    )

    objective = make_objective(BASE_EXP_ID, LOG_DIR, FID_FILE, CSV_FILE)
    csv_callback = CSVCallback(CSV_FILE)

    N_TRIALS = 200

    print(f"\n{'='*60}")
    print(f"Starting FID Optimization with Reparametrization")
    print(f"{'='*60}")
    print(f"Search space:")
    print(f"  a ∈ [0.5, 3.0]")
    print(f"  constraint (3a + b) ∈ [ln(1/4), ln(4)] ≈ [-1.386, 1.386]")
    print(f"  b ∈ [-7.0, -2.0] (constraint applied)")
    print(f"Total trials: {N_TRIALS}")
    print(f"{'='*60}\n")

    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[csv_callback])
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user.")

    print(f"\n{'='*60}")
    print(f"Optimization completed!")
    print(f"{'='*60}")
    
    if len(study.trials) > 0:
        best_trial = study.best_trial
        best_a = best_trial.user_attrs.get("a")
        best_b = best_trial.user_attrs.get("b")
        best_constraint = best_trial.user_attrs.get("constraint_value")
        
        print(f"Best trial: {best_trial.number}")
        print(f"Best FID score: {study.best_value:.4f}")
        print(f"Best parameters:")
        print(f"  a = {best_a:.4f}")
        print(f"  b = {best_b:.4f}")
        print(f"  constraint (3a + b) = {best_constraint:.4f}")
        print(f"  weights = {best_trial.user_attrs.get('weights')}")
        print(f"\nCompleted trials: {len(study.trials)}")
        print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    else:
        print("No trials completed.")
    
    print(f"\nResults saved to {CSV_FILE}")
    print(f"Study database: {DB_PATH}")
    print(f"{'='*60}")