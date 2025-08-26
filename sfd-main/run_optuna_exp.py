import subprocess
import re
import csv
import os
import optuna
from scipy.stats import uniform  # you can remove if unused

# ---------- log parsing ----------

def find_min_s2_loss(log_path: str) -> float:
    """
    Parse the log file to extract all Step 2 Loss_ls-3-mean values
    and return their minimum.
    """
    step2_ls3_losses = []
    step2_ls3_pattern = re.compile(r"Step:\s*2\s*\|\s*Loss_ls-3-mean:\s*([0-9]+\.[0-9]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = step2_ls3_pattern.search(line)
            if match:
                loss_value = float(match.group(1))
                step2_ls3_losses.append(loss_value)

    if not step2_ls3_losses:
        raise ValueError(f"No Step 2 Loss_ls-3-mean found in {log_path}")

    return min(step2_ls3_losses)

def find_last_tick(log_path: str) -> int:
    """
    Parse the log file to find the tick value from the last 'tick' line.
    """
    tick_pattern = re.compile(r"tick\s+(\d+)\b")
    last_tick = None

    with open(log_path, 'r') as f:
        for line in f:
            match = tick_pattern.search(line)
            if match:
                last_tick = int(match.group(1))

    return last_tick

# ---------- Optuna objective factory ----------

def make_objective(base_exp_id: int, log_dir: str, train_script="train.py", extra_cmd=None):
    """
    Returns an Optuna objective function which runs the external training script (subprocess),
    then parses the produced log and returns the objective (min step2 loss).
    """

    # ensure log_dir exists
    os.makedirs(log_dir, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        # Suggest hyperparameters (same ranges as original)
        a = trial.suggest_float("a", 0.0, 1.0)
        b = trial.suggest_float("b", 0.0, 1.0)
        c = trial.suggest_float("c", 0.0, 1.0)
        # lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)

        # compute weights exactly like your original script
        x = a + b + c + 1.0
        w1 = a / x
        w2 = (a + b) / x
        w3 = (a + b + c) / x
        w4 = 1.0
        weights = [w1, w2, w3, w4]

        # experiment id and folder naming (unique)
        exp_idx = trial.number + 1 + base_exp_id
        exp_folder = f"{exp_idx:05d}-cifar10-4-3-dpmpp-3-poly7.0"
        exp_path = os.path.join(log_dir, exp_folder)
        log_path = os.path.join(exp_path, "log.txt")

        print(f"[Trial {trial.number}] a,b,c = {[a, b, c]}")
        print(f"[Trial {trial.number}] weights = {weights}")
        print(f"[Trial {trial.number}] exp_folder = {exp_folder}")

        # build the command; redirect stdout/stderr to log_path so we can parse it
        cmd = [
            "python", train_script,
            "--dataset_name=cifar10",
            "--total_kimg=200",
            "--batch=128",
            f"--lr=5e-5",
            "--num_steps=4",
            "--M=3",
            "--afs=False",
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

        # allow optional extra command line args
        if extra_cmd:
            cmd.extend(extra_cmd)

        # # Run the subprocess and capture its output into log.txt
        # with open(log_path, "w") as log_file:
        #     try:
        #         subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        #     except subprocess.CalledProcessError as e:
        #         # if the training fails, mark trial as failed
        #         print(f"[Trial {trial.number}] training failed (subprocess error): {e}")
        #         # You can either return a large loss or raise to mark as failed
        #         raise
        subprocess.run(cmd, check=True)

        # after successful run, parse the log for objective
        try:
            loss = find_last_tick(log_path)
        except Exception as e:
            print(f"[Trial {trial.number}] failed to parse log: {e}")
            raise

        # store useful metadata in trial attributes
        trial.set_user_attr("exp_folder", exp_folder)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("log_path", log_path)

        # optionally report to optuna (no intermediate reporting here since training is external)
        trial.report(loss, step=0)  # final report

        print(f"[Trial {trial.number}] loss = {loss}\n")
        return loss

    return objective


# ---------- main ----------

if __name__ == "__main__":
    BASE_EXP_ID = 0
    LOG_DIR = "/home/cherish/SADD/sfd-main/exps"

    # Make an Optuna study: use TPE (Bayesian-ish) sampler + a median pruner (works if you report intermediate values)
    # If you want distributed storage use SQLite or RDB: storage="sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="diffusion_distill_weights",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()  # prune when possible (requires intermediate reports)
    )

    objective = make_objective(BASE_EXP_ID - 1, LOG_DIR)

    # config
    conf = {
        "num_iteration": 50,
        "n_jobs": 1,           # change to >1 to parallelize trials (ensure train.py can run concurrently)
    }

    try:
        study.optimize(objective, n_trials=conf["num_iteration"], n_jobs=conf["n_jobs"])
    except KeyboardInterrupt:
        print("Interrupted by user. Proceeding to save results...")

    # write CSV summary from the trials
    csv_file = "evaluation_summary_optuna.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_number", "exp_folder", "a", "b", "c", "w1", "w2", "w3", "w4", "loss", "state"])
        for t in study.trials:
            attrs = t.user_attrs
            weights = attrs.get("weights", [None, None, None, None])
            # If params exist, get a,b,c
            a = t.params.get("a")
            b = t.params.get("b")
            c = t.params.get("c")
            writer.writerow([t.number, attrs.get("exp_folder", ""), a, b, c, *weights, t.value, t.state.name])

    print("Best params:", study.best_trial.params)
    print("Best loss:", study.best_value)
    print(f"Saved all runs to {csv_file}")

