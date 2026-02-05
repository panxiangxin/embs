# embs：本地 Embedding + 游戏物品自然语言检索（JJ/NN）

本项目提供：

- `POST /v1/embeddings`：本地向量 API（默认模型 `BAAI/bge-small-zh-v1.5`）
- `POST /v1/items/import`：导入/更新物品库（支持 upsert/replace，可选触发重建索引）
- `POST /v1/index/rebuild`：全量重建索引；`POST /v1/index/refresh`：按 catalog_version 检查并刷新
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

物品管理页面（列表页 CRUD）：`http://127.0.0.1:8000/demo/types.html`

类型批量工具（重命名/合并/移除 `type` 并回写）：`http://127.0.0.1:8000/demo/type_studio.html`

## 2) 加载示例物品库

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/items/import?rebuild=true&mode=replace" `
  -H "Content-Type: application/json" `
  --data-binary "@data/items_sample.json"
```

## 3) 执行检索（debug 打开会带可解释信息/阈值相关指标）

默认使用 `jieba` 做 POS（更快，适合低延迟）；也可传 `pos_backend:"hanlp"`（更准但更慢）。
如需更稳地处理短 query（例如“红车”），可在 `config.heuristics` 调整拆分与后缀加分参数。

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
