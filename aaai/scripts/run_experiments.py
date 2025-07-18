# scripts/run_experiments.py

from brickkiln.config import all_experiments
from brickkiln.experiment import run_experiment

if __name__ == "__main__":
    for exp in all_experiments:
        print(f"\n==== Running: {exp['experiment_name']} ====")
        run_experiment(exp, device="cuda:2", epochs=50, batch_size=128, eval_every=25)

