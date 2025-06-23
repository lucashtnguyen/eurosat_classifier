## ✅ EuroSAT Land Cover Classifier

### 1 · Scope & Objectives

1. **Purpose & success metrics**
   This project builds a deep-learning image classifier using the **EuroSAT land cover dataset**, focused on demonstrating professional engineering practices: **top-down design**, **test-driven development**, and **YAML-based configuration**. It uses **PyTorch**, NumPy-style documentation, and GitHub CI to ensure clarity, reproducibility, and maintainability.
   **Success =** Reproducible training and output via a single `train(config_path: Path)` call, >90% accuracy, >85% test coverage, and clean audit logs for all runs.

2. **Core features**

* YAML-configured training, model, and data settings
* Deterministic output path: `{base_path}/{timestamp}/`
* All outputs include model weights, metrics, confusion matrix, and a copy of the full resolved config
* PyTorch-only model + training loop
* Fully modular and importable (installable library layout)
* Clean TDD with `pytest`, NumPy-style docstrings, and `black` formatting

**Definition of done:** A call to `train(pathlib.Path("path/to/config.yaml"))` launches a complete run and saves all artifacts to timestamped output folder.

3. **Non-goals**

* No Keras fallback
* No command-line interface or CLI args
* No frontend or deployment UI
* No `pyproject.toml` — installable via `setup.cfg` only

---

### 2 · Domain & Data

| Item                 | Detail                                                                                                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Primary domain       | Remote sensing / land classification                                                                                             |
| Data source          | [EuroSAT RGB](https://github.com/phelber/eurosat) or [Kaggle mirror](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) |
| Format               | \~27,000 JPGs (64×64×3), 10 labeled classes                                                                                      |
| Licensing/compliance | [EuroSAT License](https://github.com/phelber/eurosat/blob/master/LICENSE); academic use only                                     |

---

### 3 · Technical Blueprint

**Language:** Python 3.12
**Framework:** PyTorch
**Docstring Format:** NumPy-style
**Architecture:**

```
eurosat_classifier/
├── train.py                   # def train(config_path: Path)
├── config/
│   └── base_config.yaml       # Parameters for model, training, output
├── eurosat/
│   ├── __init__.py
│   ├── config.py              # YAML loader + schema validator
│   ├── data_loader.py         # Torch dataset, transforms, split logic
│   ├── model.py               # CNN builder from config
│   ├── train_loop.py          # Epoch loop, optimizer, scheduler
│   └── evaluate.py            # Accuracy, F1, confusion matrix
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_train_loop.py
├── outputs/
│   └── 2025-06-10_15-42/
│       ├── model.pt
│       ├── config.yaml         # Resolved snapshot of training config
│       ├── metrics.json
│       └── confusion_matrix.png
├── requirements.txt
├── setup.cfg
└── .github/
    └── workflows/
        └── test.yml
```

**Dependencies:**

* `torch==2.3.0`
* `torchvision==0.18.0`
* `pyyaml==6.0.1`
* `scikit-learn==1.5.0`
* `matplotlib==3.8.4`
* `pytest==8.2.2`
* `tqdm==4.66.4`
* `black==24.4.2`

---

### 4 · Coding Standards & Debug Workflow

* Use NumPy-style docstrings in **every public function and class**
* All parameters passed via `config: dict` loaded from `.yaml`
* Training logs to stdout; metrics and config are written to output dir
* Use structured logger and optional `log_buffer` for debugging
* Set `DEBUG = False` before final run

---

### 5 · Testing & Quality

**pytest roadmap:**

* `test_config.py`: required keys, type checks, default fallback
* `test_data_loader.py`: correct label count, shape, reproducibility
* `test_model.py`: forward pass, weight count, dropout presence
* `test_train_loop.py`: dry run (CPU, 1 epoch), metric logging

**Coverage Target:** ≥85% of non-GPU code

**CI Workflow (`.github/workflows/test.yml`):**

```yaml
name: Test & Lint

on: [push, pull_request]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: 3.12
      - name: Install
        run: pip install -r requirements.txt
      - name: Lint with black
        run: black --check eurosat tests
      - name: Run Tests
        run: pytest --disable-warnings
```

---

### 6 · Documentation, Packaging & Artefacts

* ✅ Use NumPy-style docstrings only (no Sphinx)
* ✅ Installable via `pip install -e .` using `setup.cfg`
* ✅ `train()` entrypoint takes a single `Path` to config
* ✅ YAML defines:

  * Model structure
  * Optimizer settings
  * Dataset path and splits
  * Output path base (`base_dir`) which expands to `{base_dir}/{timestamp}/`

**Sample `base_config.yaml`:**

```yaml
model:
  name: baseline_cnn
  hidden_units: 128
  dropout: 0.3

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 10
  seed: 42

data:
  root_dir: ./data/EuroSAT
  train_split: 0.8

output:
  base_dir: ./outputs/
```

---

### 7 · Bias, Ethics & Scientific Rigour

* Seed all randomness (dataset split, model init, dataloaders)
* Log resolved config alongside metrics for auditability
* Use stratified sampling for class balance
* All model behavior explainable via outputs (accuracy + confusion matrix)

---

### 8 · Deliverables & Timeline

| Milestone | Output                       | Owner   | ETA        |
| --------- | ---------------------------- | ------- | ---------- |
| 1         | Final spec (this document)   | ChatGPT | 2025-06-10 |
| 2         | `config.py` + test scaffold  | You     | 2025-06-12 |
| 3         | Model + training pipeline    | You     | 2025-06-15 |
| 4         | Final output & traceable run | You     | 2025-06-17 |

---
