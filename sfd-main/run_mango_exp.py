import subprocess
import re
import csv
from mango import Tuner, scheduler
from scipy.stats import uniform

evaluation_records = []

def find_min_s2_loss(log_path: str) -> float:
    """
    Parse the log file to extract the last loss values for steps 0, 1, 2
    and return their sum.
    """
    all_losses = [[] for _ in range(3)]
    pattern = re.compile(r"Step:\s*(\d+)\s*\|\s*Loss-mean:\s*([0-9]+\.[0-9]+)")
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                if 0 <= step < 3:
                    all_losses[step].append(loss)
    step_2_losses=all_losses[-1]
    # print(min(step_2_loss[-400:]))

    return min(step_2_losses[-400:])


def make_objective(base_exp_id: int, log_dir: str):
    """
    Returns an objective function that Mango can optimize.
    The function maps (a, b, c) -> structured weights with d=1 -> runs training -> returns sum of last losses.
    """
    @scheduler.serial
    def objective(a, b, c):

        x = a + b + c + 1.0
        w1 = a / x
        w2 = (a + b) / x
        w3 = (a + b + c) / x
        w4 = 1.0
        weights = [w1, w2, w3, w4]

        objective.call_count = getattr(objective, 'call_count', 0) + 1
        exp_id = base_exp_id + objective.call_count
        exp_folder = f"{exp_id:05d}-cifar10-4-3-dpmpp-3-poly7.0"
        log_path = f"{log_dir}/{exp_folder}/log.txt"

        print(f"[Experiment {exp_id}] a,b,c,d = {[a, b, c, 1.0]}")
        print(f"[Experiment {exp_id}] weights = {weights}")

        cmd = [
        "python", "train.py",
        "--dataset_name=cifar10",
        "--total_kimg=200",
        "--batch=128",
        "--lr=5e-5",
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
        f"--weight_ls={','.join(f'{w:.4f}' for w in weights)}"
        ]
        subprocess.run(cmd, check=True)

        loss = find_min_s2_loss(log_path)
        print(f"[Experiment {exp_id}] loss = {loss}\n")
        evaluation_records.append({
            'exp_folder': exp_folder,
            'weights': weights,
            'loss': loss
        })
        return loss

    return objective


if __name__ == '__main__':
    BASE_EXP_ID = 147
    LOG_DIR = "/teamspace/studios/this_studio/sfd-main/exps"

    objective = make_objective(BASE_EXP_ID - 1, LOG_DIR)

    param_space = {
        'a': uniform(0, 1),
        'b': uniform(0, 1),
        'c': uniform(0, 1)
    }

    conf = {
        'num_iteration': 50,
        'initial_random': 4,
        'domain_size': 1000
    }

    tuner = Tuner(param_space, objective, conf)
    results = tuner.minimize()

    csv_file = 'evaluation_summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_folder', 'w1', 'w2', 'w3', 'w4', 'loss'])
        for rec in evaluation_records:
            writer.writerow([rec['exp_folder'], *rec['weights'], rec['loss']])

    print("Best parameters (a,b,c):", results['best_params'])
    print("Best (lowest) loss:", results['best_objective'])
    print(f"Saved all runs to {csv_file}")
