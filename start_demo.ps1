$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\\demo\\index.html")) {
  throw "demo/index.html not found."
}

$demoUrl = $env:DEMO_URL
if (-not $demoUrl) { $demoUrl = "http://127.0.0.1:8000/demo/" }

try {
  Invoke-WebRequest -Uri $demoUrl -Method Get -TimeoutSec 1 | Out-Null
  Start-Process $demoUrl
} catch {
  Start-Process .\demo\index.html
}
