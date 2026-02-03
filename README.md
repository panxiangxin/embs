# BGE Small ZH Local API

本项目提供 `BAAI/bge-small-zh-v1.5` 的本地向量 API。

## 1) 一键启动

```powershell
.\start.ps1
```

## 2) 打开 Web Demo

```powershell
# 先确保 API 已启动
Start-Process .\demo\index.html
```

## 3) 手动创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 4) 启动服务

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 5) 调用示例

```powershell
curl -X POST "http://127.0.0.1:8000/v1/embeddings" ^
  -H "Content-Type: application/json" ^
  -d "{\"input\":[\"你好，世界\"],\"normalize\":true}"
```
