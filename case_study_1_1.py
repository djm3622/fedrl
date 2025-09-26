import numpy as np
import argparse
from utils.general import set_seed, load_config
from envs.case_study_1_1.bandits import HeteroBandit

from eval.tabular import init_summary_table, aggregate_summaries
from agents.case_study_1_1.train_local import train_local
from agents.case_study_1_1.train_fedavg import train_fedavg
from agents.case_study_1_1.train_fedtrunc import train_fedtrunc
from agents.case_study_1_1.train_fedrl import train_fedrl

from eval.metric_losses import output_metric_comparison


methods = ['local', 'fedavg', 'fedtrunc', 'fedrl']


def main(config_path: str):
    cfg = load_config(config_path, config_type='case_1')

    all_summaries = []

    for seed in cfg.seeds:
        print(f"\n--- Running with Seed: {seed} ---")
        set_seed(seed)
        env = HeteroBandit(cfg.n_clients, cfg.n_arms, np.random.default_rng(seed))

        summary = init_summary_table()  # NEW summary per seed

        for method in methods:
            print(f"Running Method: {method}")
            if method == 'local':
                summary = train_local(env, cfg, summary)
            elif method == 'fedavg':
                summary = train_fedavg(env, cfg, summary)
            elif method == 'fedtrunc':
                summary = train_fedtrunc(env, cfg, summary)
            elif method == 'fedrl':
                summary = train_fedrl(env, cfg, summary)

        all_summaries.append(summary)

    results = aggregate_summaries(all_summaries)
    output_metric_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for training environment.")
    parser.add_argument("config", type=str, help="Path to the config file.")
    
    args = parser.parse_args()
    main(args.config)