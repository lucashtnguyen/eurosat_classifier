# EuroSAT Classifier

Minimal pipeline for land cover classification using the EuroSAT dataset.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

Train with:

```python
from pathlib import Path
from train import train
train(Path("config/base_config.yaml"))
```
