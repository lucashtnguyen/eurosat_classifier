# EuroSAT Classifier
[![CI](https://github.com/lucashtnguyen/eurosat_classifier/actions/workflows/python-ci.yml/badge.svg)](https://github.com/lucashtnguyen/eurosat_classifier/actions/workflows/python-ci.yml)

Minimal pipeline for land cover classification using the EuroSAT dataset.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install
pytest --cov=eurosat --cov-fail-under=0
```

### EuroSAT dataset cache

Tests marked with `@pytest.mark.slow` require the RGB version of the EuroSAT
dataset. The data is cached under `.data/eurosat/` and will be downloaded
automatically if missing.

To fetch the dataset manually for offline work:

```bash
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='.data/eurosat', download=True)"
```

Train with:

```python
from pathlib import Path
from train import train
train(Path("config/base_config.yaml"))
```

Formatting and tests can be run automatically using pre-commit hooks:

```bash
pre-commit run --all-files
```
