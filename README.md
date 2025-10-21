# RL + Fractional Differencing for Portfolio Optimization

Minimal, reproducible implementation of **Reinforcement Learning with Fractional Differencing (FD)** for portfolio management.  
Tested on **Ubuntu 24.04**, **Python 3.12**, **PyTorch 2.9**.

---

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> `requirements.txt` includes: torch, pandas, numpy, scipy, statsmodels, gymnasium, matplotlib, tqdm, pyyaml.

## Data

Place your files under `./data/` (e.g., `prices.csv`, `universe.txt`).  
If your script auto-downloads, the folder is created on first run.

---

## Quickstart (one run per command)

**Baseline (no FD):**

```bash
python training_frac.py   --algo a2c   --mode baseline   --symbols-file data/universe.txt   --prices-csv data/prices.csv   --start 2012-01-01 --end 2024-12-31   --seed 42
```

**FD-enhanced (choose one d):**

```bash
python training_frac.py   --algo a2c   --mode fd   --d 0.3   --symbols-file data/universe.txt   --prices-csv data/prices.csv   --start 2012-01-01 --end 2024-12-31   --seed 42
```

> Keep all settings identical between baseline and FD except `--mode` and `--d`.

---

## Common Flags

`--algo {a2c,ppo,sac}` · `--mode {baseline,fd}` · `--d <0-1>` (FD only) ·  
`--symbols-file` · `--prices-csv` · `--start` · `--end` · `--seed` ·  
`--epochs` · `--lr` · `--gamma` · `--transaction-cost-bps` · `--margin` · `--shorting`

---

## Outputs

Each run creates `runs/<timestamp>_<algo>_<mode>.../` with:

- `config.yaml`, `metrics.csv`, `eval_summary.json`  
- `checkpoints/` (weights), `plots/` (PnL, drawdown, rolling Sharpe)

**Evaluate a run:**

```bash
python evaluate.py   --run-dir runs/2025-10-20_a2c_fd_d0.3_seed42   --report out/report_fd_a2c_d03.html
```

---

## Notes

- **FD regimes:** low (`d=0.2–0.4`) vs high (`d=0.7–0.9`).  
- FD can stabilize non-stationary returns while preserving long memory.

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

**License:** see `LICENSE`. • **Contact:** [your.email@example.com](mailto:your.email@example.com)
