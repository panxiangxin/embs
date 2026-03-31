# Jina-CLIP-v2 服务启动脚本
param(
    [string]$HostAddr = "0.0.0.0",
    [int]$Port = 8001,
    [string]$Device = "auto",
    [int]$TruncateDim = 1024,
    [string]$ModelName = "jinaai/jina-clip-v2"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Jina-CLIP-v2 本地 Embedding 服务启动器   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 检查 Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "未找到 Python，请先安装 Python 3.9+"
    exit 1
}

$pyVersion = & python --version 2>&1
Write-Host "Python 版本: $pyVersion" -ForegroundColor Green

# 创建虚拟环境
$venvPath = ".venv-jina-clip"
if (-not (Test-Path $venvPath)) {
    Write-Host "创建虚拟环境..." -ForegroundColor Yellow
    python -m venv $venvPath
}

# 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Yellow
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

# 升级 pip
Write-Host "升级 pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip

# 安装依赖
Write-Host "安装依赖..." -ForegroundColor Yellow
& $venvPip install -r requirements-jina-clip.txt

# 设置环境变量
$env:MODEL_NAME = $ModelName
$env:DEVICE = $Device
$env:TRUNCATE_DIM = $TruncateDim
$env:HOST = $HostAddr
$env:PORT = $Port

Write-Host "" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green
Write-Host "配置信息:" -ForegroundColor Green
Write-Host "  模型: $ModelName" -ForegroundColor White
Write-Host "  设备: $Device (auto=自动检测 CUDA/MPS/CPU)" -ForegroundColor White
Write-Host "  维度: $TruncateDim (Matryoshka 64-1024)" -ForegroundColor White
Write-Host "  地址: http://$HostAddr`:$Port" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green
Write-Host "" -ForegroundColor White

# 首次启动会下载模型（约 3.5GB）
Write-Host "注意: 首次启动会自动下载模型（约 3.5GB）到本地缓存" -ForegroundColor Yellow
Write-Host "模型缓存位置: %USERPROFILE%\.cache\huggingface\hub" -ForegroundColor Yellow
Write-Host "" -ForegroundColor White

# 启动服务
& $venvPython jina_clip_service.py
