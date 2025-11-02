# RL + Fractional Differencing for Portfolio Optimization

Minimal, reproducible A2C pipeline that **adds Fractional Differencing (FD) features** to a FinRL margin-trading environment.  
Tested on **Ubuntu 24.04**, **Python 3.12**, **PyTorch 2.9**.

FD idea in one line: we transform prices to **fractionally differenced series** (parameter `d ∈ (0,1)`), which reduces non-stationarity while keeping long-memory structure—often stabilizing RL learning signals.

---

## Project Layout

```
.
├─ env/
│  ├─ marginEnv.py      # custom environment (MarginTradingEnv)
│  └─ agent.py          # DRLAgent wrapper
├─ frac_featurs.py      # ts_differencing_tau(...) used by the script
├─ training_frac.py     # the script in this README (A2C baseline+FD features)
├─ datasets/            # created at runtime (cached csv)
└─ ../FinRL-Library/    # FinRL repo (cloned next to this project)
```

> The script imports FinRL from `../FinRL-Library` via `sys.path.append("../FinRL-Library")`. Keep that relative placement.

---

## Environment & Install

```bash
# Python 3.12 venv avoids Ubuntu 24.04 PEP 668 issues
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

python -m pip install -U pip

# Core deps (versions play nicely with Py3.12 + Torch 2.9)
pip install torch==2.9.* pandas==2.2.* numpy==2.1.* scipy==1.13.* statsmodels==0.14.*             gymnasium==0.29.* matplotlib==3.9.* tqdm==4.* pyyaml==6.*             stable-baselines3==2.3.* yfinance==0.2.* ta==0.11.*

# FinRL (use the library this script expects)
# Clone NEXT TO this repo so the relative path works:
cd ..
git clone https://github.com/AI4Finance-Foundation/FinRL-Library.git
cd -   # back to this project root
```

> If your FinRL fork has different module names, keep the relative path but adjust imports as needed.

---

## Data

- By default the script pulls **Dow 30** with `YahooDownloader` for:
  - Train: `2010-01-01 → 2019-12-31`
  - Test:  `2020-01-01 → 2021-04-30`
  - Trade: `2021-05-01 → 2024-05-01`
- A cached file is saved at `./datasets/data.csv` after first run.

To use your own data later, replace the Yahoo download block with your loader and keep the same columns FinRL expects (`date`, `tic`, `open/close/high/low/volume`, etc.).

---

## How FD Features Are Built (what the code does)

- For each ticker, the script computes **FD returns** for `d = 0.01, 0.02, ..., 0.99`:
  ```python
  d_values = [i / 100 for i in range(1, 100)]
  tau = 0.0005  # cutoff threshold inside ts_differencing_tau
  ```
- Columns are named like `d_0.20_returns`, `d_0.30_returns`, etc.
- The model actually uses a **subset** of them as indicators:
  ```python
  INDICATORS = ["d_0.20_returns", "d_0.30_returns", "d_0.40_returns"] + INDICATORS
  ```
  Change this line to switch regimes:
  - **Low-FD:** `["d_0.20_returns","d_0.30_returns","d_0.40_returns"]`
  - **High-FD:** `["d_0.70_returns","d_0.80_returns","d_0.90_returns"]`

---

## Quickstart (A2C, single run per command)

The script exposes a few A2C hyperparameters only:

- `--n_steps` (default 15)
- `--gamma`   (default 0.999)
- `--lr`      (default 0.005)
- `--ent_coef`(default 0.01)
- `--penalty` (Sharpe penalty used by env; default 0.1)

**Run with defaults:**
```bash
python training_frac.py
```

**Run with explicit hyperparameters (example):**
```bash
python training_frac.py --n_steps 15 --gamma 0.999 --lr 0.005 --ent_coef 0.01 --penalty 0.1
```

> To compare **Baseline vs FD**, run **two separate commands** after editing the `INDICATORS` line:
> 1) Baseline-ish: comment out the FD columns (use only default `INDICATORS` from FinRL)  
> 2) FD-enhanced: include your chosen `d_*_returns` columns

Keep everything else identical between runs.

---

## Outputs

The script timestamps output directories using FinRL’s `RESULTS_DIR`, e.g.:
```
results/<YYYYMMDD-HHhMM>/a2c/
```
Inside you’ll find:
- `args.txt` — saved CLI args  
- `perf_stats_test.csv`, `perf_stats_trade.csv`  
- `perf_stats_dji.csv` — DJI baseline stats for trade period  
- `test_result.csv`,  `trade_result.csv` — account value vs DJI  
- `profit_test.png`, `profit_trade.png` — equity curves  
- `test_actions.csv`, `trade_actions.csv` — actions taken  
- `test_state.csv`,   `trade_state.csv`  — state traces  
- `../trained_models/<timestamp>/agent_a2c.pth` — weights

---

## Reproducibility Tips

- Seed is fixed at the top:
  ```python
  seed = 0
  set_random_seed(seed)
  ```
- To make FD regime changes explicit, commit the exact `INDICATORS` list you used.
- When comparing, don’t mix hyperparameters: change **only** the FD indicators.

---

## Notes

- **Why FD helps (intuition):** raw price series are non-stationary; RL rewards built on such signals can be noisy and drift-y. FD reduces unit-root behavior while preserving long-range dependence, which can improve learning stability.
- **Transaction costs:** set via `buy_cost_pct`/`sell_cost_pct` in `env_kwargs` (here `0.1%`).  
- **Margin / shorting:** handled by `MarginTradingEnv` (see `env/marginEnv.py`); adjust there if you need different leverage rules.

---

## Cite

**This work (edit fields):**
```bibtex
@article{fd_rl_2025,
  title   = {Reinforcement Learning Enhanced by Fractional Differencing for Portfolio Optimization},
  author  = {<Your Name>},
  journal = {<Venue or Preprint>},
  year    = {2025}
}
```

**Margin Trader (ICAIF 2023):**
```bibtex
@inproceedings{gu2023margin,
  title={Margin Trader: A Reinforcement Learning Framework for Portfolio Management with Margin and Constraints},
  author={Gu, Jingyi and Du, Wenlu and Rahman, AM Muntasir and Wang, Guiling},
  booktitle={Proceedings of the Fourth ACM International Conference on AI in Finance},
  pages={610--618},
  year={2023}
}
```

---

## License & Contact

- **License:** see `LICENSE`.  
- **Contact:** your.email@example.com

---

# Next: expose FD/Algo via CLI (optional)
If you later want true CLI control for FD & algorithm, add flags like `--algo`, `--mode baseline|fd`, and `--fd-cols ...`, and construct the `INDICATORS` list from those args.
