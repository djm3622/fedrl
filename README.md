# Federated Distributional Reinforcement Learning (FedRL-Main)

This repository contains research code for **federated distributional reinforcement learning** with multiple training modes and two case studies:
1) a heterogeneous **bandit** benchmark (Case Study 1.1), and  
2) a **multi‑agent gridworld** with hazards and partial observability (Case Study 2.1).

The codebase supports **Local**, **FedAvg (critic‑only)**, **FedTrunc (critic w/o head)**, and **FedRL (hazard‑weighted trust‑region)** training modes. Logging and artifact tracking are integrated via Weights & Biases (optional).

---

## Table of Contents
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Case Study 1.1 — Bandits](#case-study-11--bandits)
  - [Case Study 2.1 — Multi‑Agent Gridworld](#case-study-21--multiagent-gridworld)
- [Training Modes](#training-modes)
- [Configuration Guide](#configuration-guide)
  - [Common Federation Settings](#common-federation-settings)
  - [Case Study 1.1 Config](#case-study-11-config)
  - [Case Study 2.1 Config](#case-study-21-config)
  - [FedRL‑Specific Settings](#fedrlspecific-settings)
- [Outputs and Logging](#outputs-and-logging)

---

## Repository Layout

```
fedrl-main/
├─ README.md                        # short placeholder (this README is the comprehensive version)
├─ LICENSE
├─ case_study_1_1.py                # entrypoint for bandit case
├─ case_study_2_1.py                # entrypoint for gridworld case (with CLI --method)
├─ configs/
│  ├─ case_study_1_1.yaml           # example config for bandits
│  ├─ case_study_2_1.yaml           # example config for gridworld
│  └─ config_templates/
│     ├─ case_study_1_1.py          # dataclass template for case 1.1 config
│     └─ case_study_2_1.py          # dataclass template for case 2.1 config
├─ envs/
│  ├─ case_study_1_1/               # bandit environment
│  └─ case_study_2_1/               # gridworld environment(s)
├─ models/
│  ├─ c51_bandit.py                 # bandit distributional model
│  ├─ fedrl_nets.py                 # shared nets/utilities
│  └─ mappo_nets.py                 # MAPPO actor/critic networks for gridworld
├─ agents/
│  ├─ case_study_1_1/
│  │  ├─ train_local.py
│  │  ├─ train_fedavg.py
│  │  ├─ train_fedtrunc.py
│  │  └─ train_fedrl.py
│  └─ case_study_2_1/
│     ├─ ppo_utils/                 # PPO buffer, helpers, trainer
│     ├─ train_local.py
│     ├─ train_fedavg.py
│     ├─ train_fedtrunc.py
│     └─ train_fedrl.py
├─ federation/
│  └─ case_study_2_1/
│     ├─ base.py
│     ├─ fedavg.py                  # critic-only aggregation
│     ├─ fedtrunc.py                # critic minus head aggregation
│     ├─ fedrl.py                   # hazard-weighted + trust-region blend
│     └─ barycenter.py              # experimental reward fusion utilities
├─ setup/
│  ├─ requirements_case_study_1_1.txt
│  └─ requirements_case_study_2_1.txt
└─ utils/
   ├─ general.py                    # config loading, seeding, helpers
   └─ wandb_helper.py               # Weights & Biases utilities
```

---

## Requirements

Create a Python environment (3.9–3.11 recommended), then install dependencies per study:

```bash
# Case Study 1.1 (bandits)
pip install -r setup/requirements_case_study_1_1.txt

# Case Study 2.1 (gridworld + PPO/MAPPO)
pip install -r setup/requirements_case_study_2_1.txt
```

---

## Quick Start

### Case Study 1.1 — Bandits

Run all methods for a set of seeds as specified in the config:

```bash
python case_study_1_1.py configs/case_study_1_1.yaml
```

Artifacts and summaries will be written to the output folders configured via W&B settings (optional) and local paths.

**Key files used:**
- `envs/case_study_1_1/bandits.py`
- `models/c51_bandit.py`
- `agents/case_study_1_1/train_{local,fedavg,fedtrunc,fedrl}.py`

---

### Case Study 2.1 — Multi‑Agent Gridworld

Choose a method with `--method {local,fedavg,fedtrunc,fedrl}` and a config:

```bash
# Local training for N clients (single-process baseline)
python case_study_2_1.py --config configs/case_study_2_1.yaml --method local

# FedAvg (critic-only aggregation)
python case_study_2_1.py --config configs/case_study_2_1.yaml --method fedavg

# FedTrunc (aggregate critic except the distributional head)
python case_study_2_1.py --config configs/case_study_2_1.yaml --method fedtrunc

# FedRL (hazard-weighted + server trust-region blending)
python case_study_2_1.py --config configs/case_study_2_1.yaml --method fedrl
```

On the first run, the script saves model architecture snapshots to:
```
outputs/<wandb_project>/<wandb_run_name>/{actor_arch.txt,critic_arch.txt}
```

---

## Training Modes

- **Local**  
  Independent training per client; no parameter sharing.

- **FedAvg (critic‑only)**  
  Aggregates only the **critic** parameters across clients (actor/policy remains local).  
  See `federation/case_study_2_1/fedavg.py`. This matches the code’s packing/unpacking of `"critic.*"` keys.

- **FedTrunc (critic w/o head)**  
  Aggregates critic parameters **excluding** the final distributional head. This keeps the quantile/logit head local while sharing the backbone.  
  See `federation/case_study_2_1/fedtrunc.py`.

- **FedRL (barycenter trust‑region prior)**
  Ignores parameter federation and instead builds a per-round barycenter over the unique state distributions encountered (frequency- and CVaR-weighted). The barycenter is broadcast as a frozen critic prior for the next round while preserving the existing shrinkage + clamp logic.
  See `agents/case_study_2_1/train_fedrl.py` and `federation/case_study_2_1/barycenter.py` for details.

---

## Configuration Guide

Configuration files are standard YAML and are validated against dataclass templates in `configs/config_templates/`. Loaders in `utils/general.py` map:
- `case_study_1_1.yaml` → `Config` (bandits)
- `case_study_2_1.yaml` → `MultiAgentGridConfig` (gridworld)

Edit the provided YAMLs to change devices, seeds, federation schedule, PPO hyperparameters, grid layout, starts/goals, and logging.

### Common Federation Settings

Both studies expose:
- `federation.num_communication_rounds` — global rounds
- `federation.local_epochs_per_round` — epochs of on‑client training per round
- `federation.n_clients` — number of clients
- `federation.client_weights` — nonnegative weights; normalized by volume (see `utils/general.py::norm_vol_weights`)
- `federation.cvar_alpha` — risk metric level (if used downstream)

### Case Study 1.1 Config

File: `configs/case_study_1_1.yaml` (selected keys)
```yaml
model:
  device: "mps"            # "cpu" | "cuda" | "mps"
  lr: 1e-4
  batch_size: 1
  grad_clip: 10.0
  seeds: [2, 3, 4, 5]

federation:
  num_communication_rounds: 1000
  local_epochs_per_round: 2
  n_clients: 5

# ... environment and reward settings omitted for brevity ...

eval:
  cvar_alpha: 0.1
  client_weights: [0.8, 0.05, 0.05, 0.05, 0.05]

fedrl:
  kappa: 0.05
  server_kd_lr: 1e-1
  ot_reg: 1e-1
  ot_p: 1
  server_kd_steps: 10
```

### Case Study 2.1 Config

File: `configs/case_study_2_1.yaml` (selected keys)
```yaml
train:
  device: "mps"            # "cpu" | "cuda" | "mps"
  total_steps: 10000000
  rollout_len: 1024
  update_epochs: 4
  minibatches: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.05
  critic_target_ema: 0.01
  lr_actor: 0.0001
  lr_critic: 0.0002
  # ...

federation:
  num_communication_rounds: <int>
  local_epochs_per_round: <int>
  n_clients: <int>
  client_weights: [ ... ]  # will be normalized

env:
  H: 10
  W: 10
  # starts/goals/obstacles/templates and other layout controls

wandb:
  wandb_project: "mappo-gridworld"
  wandb_run_name: "case_study_2_1_example"
  wandb_mode: "online"     # "disabled" to turn off
  log_video: false

eval:
  cvar_alpha: 0.1
  client_weights: [0.8, 0.05, 0.05, 0.05, 0.05]

fedrl:
  kappa: 0.05
  server_kd_lr: 1e-1
  ot_reg: 1e-1
  ot_p: 1
  server_kd_steps: 10
```

> The full set of fields (grid layout, object goals, templates, etc.) is defined in `configs/config_templates/case_study_2_1.py`. Review that file for all available options and their validation rules.


## Outputs and Logging

- **Local artifacts**  
  `outputs/<wandb_project>/<wandb_run_name>/`
  - `actor_arch.txt`, `critic_arch.txt` — textual architecture dumps for provenance.
  - Model checkpoints are saved by the individual training runners (see `agents/.../train_*.py`).

- **Weights & Biases (optional)**  
  Use `wandb_mode: "online"` with a valid login to:
  - Log metrics (returns, hazards, losses).
  - Upload architecture text files as artifacts.
  - Track per‑round aggregation weights and diagnostics.

To disable W&B entirely, set `wandb_mode: "disabled"` in the config.
