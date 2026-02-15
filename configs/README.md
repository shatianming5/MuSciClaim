# configs/

YAML configs for models and runs. These are the primary knobs for reproducible evaluations.

## Contents

| Path | Role | Why it exists |
| --- | --- | --- |
| `configs/README.md` | Config index | Explains config files and reproducibility conventions |
| `configs/models.example.yaml` | Example model configuration | Template you copy to `configs/models.yaml` |
| `configs/run.example.yaml` | Example run configuration | Template you copy to `configs/run.yaml` |
| `configs/models.yaml` | Your model IDs and revisions (local) | Keeps model provenance explicit and reproducible (gitignored by default) |
| `configs/run.yaml` | Your run matrix and IO settings (local) | Controls dataset revision, preprocessing, and evaluation matrix (gitignored by default) |
