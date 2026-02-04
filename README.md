# embs：本地 Embedding + 游戏物品自然语言检索（JJ/NN）

本项目提供：

- `POST /v1/embeddings`：本地向量 API（默认模型 `BAAI/bge-small-zh-v1.5`）
- `POST /v1/items/load`：加载物品库并构建索引（BM25 + 向量 + 负样本分布）
- `POST /v1/item_search`：基于 POS（JJ/NN）拆分的物品检索，并输出 `ACCEPT/CLARIFY/REJECT`

## 1) 一键启动

```powershell
.\start.ps1
```

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

## 1.1) 打开 Web 页面

两种方式任选其一：

```powershell
# 方式 A：由 API 服务静态托管（推荐）
Start-Process "http://127.0.0.1:8000/demo/"

# 方式 B：直接打开本地文件
Start-Process .\demo\index.html
```

## 2) 加载示例物品库

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/items/load" `
  -H "Content-Type: application/json" `
  --data-binary "@data/items_sample.json"
```

## 3) 执行检索（debug 打开会带可解释信息/阈值相关指标）

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/item_search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"破损的蓝色卡车\",\"debug\":true}"
```

只在“候选集合”里搜索（例如背包/附近物品 id 列表）：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/item_search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"红色的破烂东西\",\"candidate_ids\":[\"veh-01\",\"str-01\"],\"debug\":true}"
```

## 4) 本地自测脚本

```powershell
python .\scripts\smoke_test_item_search.py
```

## 5) Embedding API 示例

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/embeddings" `
  -H "Content-Type: application/json" `
  -d "{\"input\":[\"你好，世界\"],\"normalize\":true}"
```
