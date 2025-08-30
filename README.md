<div align="center">

# Tetris AI  
_Reinforcement‑learning agent for **turn‑based TETR.IO** (self‑play + MCTS/PUCT)_

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Backend](https://img.shields.io/badge/Backends-Keras%2FTFLite%20%7C%20PyTorch-6aa84f)](#)
[![Status](https://img.shields.io/badge/status-experimental-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)](#)

</div>

---

## 🎥 Demo


https://github.com/user-attachments/assets/86f91867-77d1-4d46-bd22-3bae9afb2d26


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

# 2) Install dependencies from requirements.txt
pip install -r requirements.txt

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
    model='keras',
    use_tflite=True,
    model_version=5.9,
    data_version=2.4,
    ruleset='s2',
    blocks=10,
    filters=16,
    dropout=0.25,
    l2_reg=3e-5,
    o_side_neurons=16,
    value_head_neurons=16,
    MAX_ITER=160,
    CPUCT=0.75,
    DPUCT=1,
    FpuStrategy='reduction',
    FpuValue=0.4,
    use_root_softmax=True,
    RootSoftmaxTemp=1.1,
    use_dirichlet_noise=True,
    DIRICHLET_ALPHA=0.02,
    DIRICHLET_S=25,
    DIRICHLET_EXPLORATION=0.25,
    use_dirichlet_s=True,
    training=False,
    learning_rate=1e-3,
    loss_weights=[1, 1],
    epochs=1,
    batch_size=64,
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
├─ requirements.txt    # Dependencies
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

This project is licensed under the **MIT License**

```text
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
