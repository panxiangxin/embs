$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\\demo\\index.html")) {
  throw "demo/index.html not found."
}

Start-Process .\demo\index.html
