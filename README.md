# EA Traffic Control — Project Summary

Short overview
- This project implements an evolutionary (Differential Evolution) traffic‑signal optimizer and an MLP controller pipeline for SUMO simulations. It runs cycle‑level optimization, logs per‑cycle metrics, and trains a neural controller from DE-produced data for fast inference.

Key ideas
- Use DE to optimize 2 parameters: green split s and cycle length C, measured by two objectives: average delay (O1) and fairness (O2). The optimizer uses a score function (weighted sum) to rank candidates; e.g. $score = w_1 O_1 + w_2 O_2$ (weights configured in the code — see [`controller.score_function`](controller.ipynb)).
- Collect per‑cycle inputs/outputs to produce a teacher dataset for an MLP, then train a model for fast, on‑line control instead of running an optimizer each cycle.

Repository layout (important files)
- [controller.ipynb](controller.ipynb) — interactive driver and visualization (DE run examples, plotting utilities, score function). See [`controller.score_function`](controller.ipynb).
- [mlp_controller.ipynb](mlp_controller.ipynb) — runs the MLP controller experiments and visualizations.
- [mlp_training.ipynb](mlp_training.ipynb) — notebook to run training (calls training entrypoint).
- src/
  - [src/config.py](src/config.py) — SUMO paths, seeds, logging dirs (e.g. [`src.config.SUMO_CFG`](src/config.py), [`src.config.LOG_DIR`](src/config.py)).
  - [src/de_experiment.py](src/de_experiment.py) — DE experiment entrypoints:
    - [`src.de_experiment.seed_de_simulation`](src/de_experiment.py)
    - [`src.de_experiment.seed_de_teacher_simulation`](src/de_experiment.py)
    - [`src.de_experiment.seed_mlp_controller_simulation`](src/de_experiment.py)
    - [`src.de_experiment.baseline_simulation`](src/de_experiment.py)
  - [src/de_optimizer.py](src/de_optimizer.py) — DE mechanics and helpers.
  - [src/simulation_libs.py](src/simulation_libs.py) — SUMO wrappers, e.g. [`src.simulation_libs.find_avg_halt_range`](src/simulation_libs.py).
  - [src/traffic_metrics.py](src/traffic_metrics.py) — metrics and cycle evaluation helpers.
  - [src/road_config.py](src/road_config.py) — intersection metadata (e.g. [`src.road_config.TL_ID`](src/road_config.py), lane groups).
  - [src/utils.py](src/utils.py) — logging/visualization utilities:
    - [`src.utils.log_cycle_result`](src/utils.py)
    - [`src.utils.visualize_pow_results`](src/utils.py)
    - [`src.utils.compare_logs_by_time`](src/utils.py)
    - [`src.utils.compare_score_best`](src/utils.py)
  - [src/train_de_mlp.py](src/train_de_mlp.py) — training entrypoint: [`src.train_de_mlp.train_de_ml`](src/train_de_mlp.py).
  - [src/logging_ml_teacher.py](src/logging_ml_teacher.py) — dataset writer for MLP teacher data.
- datasets/
  - [datasets/de_teacher_data.csv](datasets/de_teacher_data.csv) — example teacher data created by DE teacher runs.
- logs/ — per‑run CSV summary files (e.g. traffic_DE_summary_*.csv, traffic_baseline_summary_*.csv).
- models/ — trained MLP models saved here.

Quick start (run locally)
1. Configure SUMO and scenario paths in [`src/config.py`](src/config.py) and intersection lanes in [`src/road_config.py`](src/road_config.py).
2. Run a DE experiment (notebook or function):
   - Open [controller.ipynb](controller.ipynb) and run the DE cells, or call [`src.de_experiment.seed_de_simulation`](src/de_experiment.py).
3. Produce teacher data (DE + logging):
   - Call [`src.de_experiment.seed_de_teacher_simulation`](src/de_experiment.py) to generate teacher CSV(s) in `datasets/`.
4. Train MLP:
   - Use [mlp_training.ipynb](mlp_training.ipynb) or call [`src.train_de_mlp.train_de_ml`](src/train_de_mlp.py).
5. Evaluate / run MLP controller:
   - Use [mlp_controller.ipynb](mlp_controller.ipynb) or [`src.de_experiment.seed_mlp_controller_simulation`](src/de_experiment.py) to test the learned policy in SUMO.
6. Visualize and compare:
   - Use functions in [`src.utils`](src/utils.py): [`src.utils.visualize_pow_results`](src/utils.py), [`src.utils.compare_logs_by_time`](src/utils.py), and [`src.utils.compare_score_best`](src/utils.py) to inspect convergence, per‑cycle metrics, and baseline vs DE/MLP comparisons.

Notes & tips
- The score used for DE is a linear combination of O1 and O2 (weights visible in the notebooks). See [`controller.score_function`](controller.ipynb) for exact implementation.
- SUMO state snapshots are used to ensure identical evaluation starts; check [`src.config.SUMO_STATE`](src/config.py) and the snapshot helpers in [`src.simulation_libs.find_avg_halt_range`](src/simulation_libs.py).
- Logs and summary CSVs are stored in `logs/` and aggregated by [`src.utils.log_cycle_result`](src/utils.py). Use those to reproduce plots shown in the notebooks.
- If you want to change the optimization bounds or DE hyperparameters, edit [`src/de_optimizer.py`](src/de_optimizer.py) or the call sites in [`src.de_experiment`](src/de_experiment.py).

References (openable files/functions)
- Files: [controller.ipynb](controller.ipynb), [mlp_controller.ipynb](mlp_controller.ipynb), [mlp_training.ipynb](mlp_training.ipynb)
- Config & metadata: [`src.config`](src/config.py), [`src.road_config`](src/road_config.py)
- DE & experiments: [`src.de_experiment`](src/de_experiment.py), [`src.de_optimizer`](src/de_optimizer.py)
- SUMO helpers: [`src.simulation_libs.find_avg_halt_range`](src/simulation_libs.py)
- Metrics & logging: [`src.traffic_metrics`](src/traffic_metrics.py), [`src.utils.visualize_pow_results`](src/utils.py)
- Training: [`src.train_de_mlp.train_de_ml`](src/train_de_mlp.py), [`src.logging_ml_teacher`](src/logging_ml_teacher.py)
- Data & outputs: [datasets/de_teacher_data.csv](datasets/de_teacher_data.csv), [logs/](logs/), [models/](models/)

License & next steps
- Add a LICENSE if you plan to publish.
- Next: tune DE hyperparameters, enrich teacher data, try alternative network architectures and validate generalization on different traffic mixes.
