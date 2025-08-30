<div align="center">

# Tetris AI  
_Reinforcement‑learning agent for **turn‑based TETR.IO** (self‑play + MCTS/PUCT)_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Backend](https://img.shields.io/badge/Backends-Keras%2FTFLite%20%7C%20PyTorch-6aa84f)](#)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)](#)

</div>

---

## 🎥 Demo

https://github.com/user-attachments/assets/0d98cd95-5b2b-4534-aa58-2d029f14011a

> The video shows coordinated play *and* a failure mode: the agent can fall into a **local minimum** that’s **exploitable** by specific setups. See **Research notes** for mitigations.

---

## ✨ Overview

**Tetris AI** is a compact research project that trains an agent to play **turn‑based TETR.IO** via self‑play and an AlphaZero‑style loop:
- Residual CNN with **policy** & **value** heads (see `architectures.py`)
- **MCTS/PUCT** search with Dirichlet noise at the root
- **Self‑play → Train → Gate** loop to promote stronger models
- Optional **TFLite** inference for speed (Keras backend)
- Basic visualization using **pygame** when `visual=True`

It’s intentionally small and hackable—ideal for experiments and coursework.

---

## 🚀 Quick start

```bash
# 1) (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps (adjust to your environment)
pip install numpy pandas matplotlib pygame
pip install tensorflow              # or: pip install tensorflow[and-cuda]
# (Optional alt backend)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) Run the simulator / training loop
python simulation.py
```

On first run, a fresh network is created if none exists. The loop alternates between self‑play data generation, training, and head‑to‑head gating.

---

## ⚙️ Configuration (cheat‑sheet)

Most behavior is controlled by a `Config` in **`ai.py`**. Example:

```python
from ai import Config, instantiate_network, self_play_loop

cfg = Config(
    # I/O & backend
    model='keras',            # 'keras' | 'pytorch'
    use_tflite=True,          # (keras only) fast inference
    model_version=5.9,        # Storage/models/<ruleset>.<model_version>
    data_version=2.4,         # Storage/data/<ruleset>.<data_version>
    ruleset='s2',             # 's1' | 's2'

    # Network (residual CNN)
    blocks=10,
    filters=16,
    dropout=0.25,
    l2_reg=3e-5,
    o_side_neurons=16,
    value_head_neurons=16,

    # MCTS / PUCT
    MAX_ITER=160,
    CPUCT=0.75,
    DPUCT=1,
    FpuStrategy='reduction',  # or 'absolute'
    FpuValue=0.4,
    use_root_softmax=True,
    RootSoftmaxTemp=1.1,

    # Exploration
    use_dirichlet_noise=True,
    DIRICHLET_ALPHA=0.02,
    DIRICHLET_S=25,
    DIRICHLET_EXPLORATION=0.25,
    use_dirichlet_s=True,

    # Training
    training=False,
    learning_rate=1e-3,
    loss_weights=[1, 1],      # [value, policy]
    epochs=1,
    batch_size=64,

    # Data
    data_loading_style='merge',
    augment_data=True,
    shuffle=True
)

instantiate_network(cfg, show_summary=True, save_network=True)
self_play_loop(cfg, skip_first_set=False)
```

> **Storage layout:** models in `Storage/models/<ruleset>.<model_version>/`, data in `Storage/data/<ruleset>.<data_version>/`.

---

## 🗂️ Project structure

```
.
├─ ai.py               # RL loop, MCTS/PUCT, training/gating, backends, Config
├─ architectures.py    # Residual CNN; policy/value heads (Keras & PyTorch)
├─ simulation.py       # Entry point (run this)
├─ util.py             # Utilities (profiling, experiments, test boards)
└─ README.md
```

---

## 🧭 Research notes (limitations & mitigations)

**Observed issue:** convergence to brittle strategies (clever but exploitable local minima).

**Mitigations available in‑code:**
- Stronger root exploration: tune `DIRICHLET_ALPHA`, `DIRICHLET_EXPLORATION`, `use_dirichlet_s`.
- **Forced playouts & policy‑target pruning:** enable `use_forced_playouts_and_policy_target_pruning=True` and set `CForcedPlayout`.
- Diversify openings: `use_random_starting_moves=True` or curriculum openings.
- Adjust FPU: `FpuStrategy`/`FpuValue`; try `use_root_softmax=True` and `RootSoftmaxTemp`.
- Optional light **reward shaping** during data generation (e.g., penalize covered holes).

---

## 📝 Notes

- This is an **experimental** project—expect rough edges.
- Full results and design details live in the **Wiki**.
- The demo intentionally highlights both competence and exploitable failure modes to guide future work.

---

## 🤝 Contributing

Issues and PRs are welcome. Small, focused changes with a brief note or demo GIF are ideal.

---

## 📄 License

Add your preferred open‑source license (MIT/Apache‑2.0/BSD‑3‑Clause) to the repo.
