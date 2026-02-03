$ErrorActionPreference = "Stop"

$venvPath = ".\\.venv"
if (-not (Test-Path $venvPath)) {
  python -m venv .venv
}

& "$venvPath\\Scripts\\python.exe" -m pip install -U pip
& "$venvPath\\Scripts\\python.exe" -m pip install -r requirements.txt

if (-not $env:MODEL_NAME) { $env:MODEL_NAME = "BAAI/bge-small-zh-v1.5" }
if (-not $env:DEVICE) { $env:DEVICE = "cpu" }

& "$venvPath\\Scripts\\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
