install:
  pip-sync requirements.txt
  python -m pip install --editable .

lock:
  pip-compile --upgrade --output-file=requirements.txt pyproject.toml

venv:
  #!/usr/bin/env bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip pip-tools