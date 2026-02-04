# 技术开发方案：游戏物品管理 + 语义物品检索 API

本文档面向现有 `embs` 项目（FastAPI + 本地 embedding + BM25 + POS(JJ/NN) 拆分），目标是把它收敛成**“物品管理 + 自然语言→物品检索”**的高性能本地服务；尽量不引入额外中间件，便于在游戏服务器或工具链中直接部署与调用。

---

## 0. 目标与边界

### 0.1 目标

- 提供**物品管理**：维护物品基础信息（name/type/aliases/desc_labels/属性），支持导入导出。
- 提供**语义物品检索 API**：将玩家自然语言描述映射到游戏内物品（或候选集合），并输出 `ACCEPT/CLARIFY/REJECT` 决策与可解释信息，用于 NPC/交互逻辑执行“找某个物品”的动作。
- 强性能：冷启动可控、查询低延迟（典型 < 30ms~80ms CPU，视物品量级而定）、可水平扩展（进程级）。
- 少中间件：默认不依赖外部 DB/Redis/ES；通过**SQLite（可选）或文件存储**实现持久化。

### 0.2 非目标（明确不做/后置）

- 不做完整的 NPC 行为树/对话系统；本服务只输出**可执行的结构化检索结果**，由游戏端决定如何驱动 NPC。
- 不做复杂工作流中间件（MQ、分布式缓存、搜索集群）作为默认架构；仅作为可选扩展。
- 不做全量权限体系（RBAC 多租户）作为第一阶段；仅提供最小的鉴权与隔离策略。

---

## 1. 典型使用场景

1) **游戏运行时**：玩家说“帮我找一下破损的蓝色卡车”，游戏端把 query + 附近候选物品 id 列表发给服务，服务返回 top1 匹配及置信度；若 `CLARIFY`，返回候选列表让 UI/NPC 追问。

2) **编辑器/运营工具**：策划或工具链通过 API 维护物品，批量导入、增量更新后触发索引重建。

3) **离线校准**：根据真实日志（query→最终选择的 item_id）调整阈值、同义词表、负样本分布参数，提升 `ACCEPT/CLARIFY/REJECT` 的可靠性。

---

## 2. 总体架构（最小依赖版本）

单体服务即可满足需求：

- **HTTP 服务层**：FastAPI + Uvicorn
- **存储层（可插拔）**
  - 默认：`file`（JSON/NDJSON）——最少依赖，适合小规模与快速迭代
  - 推荐：`sqlite`（单文件 + WAL）——无需外部中间件，支持并发读写、事务与版本化
- **索引层（内存）**
  - BM25：字面匹配召回
  - 向量检索：语义召回（默认 exact：矩阵乘法；可选 ANN：HNSW/Faiss）
  - 负样本分布/阈值：将相似度映射为误匹配概率 `p_spurious`，产出可解释的决策
- **查询解析层**
  - normalize：归一化、同义词映射
  - POS：jieba（默认）/hanlp（可选）

> 建议将“管理 API（写多）”与“查询 API（读多）”在逻辑上隔离：同进程也可通过路由前缀和最小鉴权隔离；若未来需要极致性能，可拆成两个进程（同一存储，查询进程只读）。

---

## 3. 数据模型设计

### 3.1 Item（物品）

- `item_id: str`：稳定 id（如 `veh-01`）
- `name: str`：显示名（如“蓝色卡车”）
- `type: str | list[str]`：类型/类别（如“卡车”“车”“汽车”；用于 NN 文本拼接与 BM25/向量召回）
- `aliases: list[str]`：物品别名（如“货车”）
- `desc_labels: list[str]`：描述标签（用于 JJ 覆盖与特征）
- `attrs: dict[str, Any]`：扩展属性（如 `{"color":"blue","durability":"broken"}`）
- `status: str`：`active|disabled|deleted`（软删便于回滚）
- `version: int`、`updated_at: int`

### 3.2 归一化与同义词

保持项目当前策略：`normalize(text) -> normalized_text`，并维护：

- `DEFAULT_SYNONYMS`：常用同义映射（“货车→卡车”）
- `DOMAIN_SYNONYMS`：项目/游戏专属同义（可从配置或管理 API 写入）

建议支持同义词的**版本化**与热更新：更新后触发“索引增量重建”。

---

## 4. 存储方案（少中间件）

### 4.1 file 模式（默认最少依赖）

- 数据落盘：`data/items.json`（可选：`data/synonyms.json`）
- 优点：实现快、部署简单
- 缺点：并发写入与版本控制较弱；大量更新时需要全量写回
- 适用：研发期、单机工具、物品量级较小（< 5 万）

### 4.2 sqlite 模式（推荐生产默认）

使用 SQLite 单文件（例如 `data/embs.db`），开启 WAL：

- 优点：无外部中间件、事务可靠、并发读强、支持增量更新与版本查询
- 缺点：跨机器共享较弱（但对“本地服务”定位足够）

表建议：

- `items(item_id PK, name, type_json, aliases_json, desc_labels_json, attrs_json, status, version, updated_at)`
- `synonyms(key PK, value_json, version, updated_at)`（可选）
- `embeddings(entity, entity_id, vec_blob, dim, version, updated_at)`（可选：若要持久化向量）

向量持久化的推荐策略：

- 小规模：直接存 SQLite BLOB（实现简单）
- 中大规模：用 `.npy/.npz` 或 `np.memmap` 文件持久化矩阵（高性能），SQLite 只存元信息与 row 映射

---

## 5. 索引与检索设计（高性能）

### 5.1 预处理与字段策略

为每个 item 构造“可检索文本视图”：

- `text_nn`（名词相关）：`name + type + aliases`
- `text_jj`（形容词/描述）：`desc_labels + attrs(可转标签)`

统一归一化后用于：

- BM25：对 `text_nn` 建索引（必要时对 `text_jj` 也建轻量索引）
- Embedding：分别对 `text_nn`、`text_jj` 建向量（可共享或分开）

### 5.2 召回：BM25 + 向量（混合）

推荐“粗召回→精排”两段式：

1) **粗召回 TopK**
   - BM25 TopK（专名、编号、缩写、精确命中强）
   - Vector TopK（同义/语义强）
   - 合并去重得到候选集合 C（例如 100~300）
2) **精排**
   - 采用项目现有 JJ/NN 拆分与覆盖策略：`S_total = α * S_nn + β * S_jj`
   - 用负样本分布将相似度映射为 `p_spurious`，并输出 `ACCEPT/CLARIFY/REJECT`

### 5.3 决策输出（供 NPC/交互使用）

保持当前三态，并扩展为可执行结构：

- `ACCEPT`：返回 `item_id` 与置信度；游戏端可直接驱动 NPC 去交互
- `CLARIFY`：返回候选列表（可附带“区分性特征”提示，如颜色/类型差异），由 UI/NPC 追问
- `REJECT`：返回建议（如“附近没有符合条件的物品”或提示玩家换个说法）

### 5.4 候选集合约束（强烈建议）

游戏运行时几乎总存在上下文（背包、附近、容器内、任务道具集合）。建议 `item_search` 支持：

- `candidate_ids: list[str] | None`：限定检索范围（极大提升精度与速度）
> 若需要“只在某类物品里找”的能力，建议由游戏端/工具端先做集合裁剪（按类型/标签/属性筛出 `candidate_ids`），再调用 `item_search`；服务端保持请求体最小化，仅接收 `candidate_ids` 作为范围约束。

---

## 6. API 设计（只保留管理 + 查询）

API 建议版本前缀：`/v1`

### 6.1 健康与信息

- `GET /health`：返回服务状态、模型是否就绪、索引版本
- `GET /v1/status`：返回当前配置摘要（model/device/backend/index sizes）

### 6.2 物品管理（Item）

- `POST /v1/items`：创建物品
- `PUT /v1/items/{item_id}`：更新物品
- `GET /v1/items`：列表（支持分页；可选按关键词/状态过滤）
- `GET /v1/items/{item_id}`：详情
- `DELETE /v1/items/{item_id}`：删除/禁用（软删）

批量能力（强烈建议保留，便于工具链）：

- `POST /v1/items/import`：导入（JSON/NDJSON；支持 upsert）
- `GET /v1/items/export`：导出（用于版本控制/回滚）

### 6.3 索引管理（Index）

- `POST /v1/index/rebuild`：全量重建（管理端调用）
- `POST /v1/index/refresh`：增量刷新（根据 version/updated_at）
- `GET /v1/index/info`：索引规模、版本、最近构建耗时

### 6.4 语义检索（Search）

- `POST /v1/item_search`

请求体建议：

```json
{
  "query": "破损的蓝色卡车",
  "candidate_ids": ["veh-01", "veh-02"],
  "top_k": 10,
  "debug": false,
  "pos_backend": "jieba"
}
```

响应体建议（示意）：

```json
{
  "decision": "ACCEPT",
  "item_id": "veh-01",
  "score": 0.82,
  "p_spurious": 0.006,
  "candidates": [
    {"item_id":"veh-01","score":0.82},
    {"item_id":"veh-02","score":0.64}
  ],
  "clarify_question": null,
  "debug": {
    "parsed": {"nn":["卡车"],"jj":["破损","蓝色"],"head_noun":"卡车"},
    "explain": {"s_nn":0.78,"s_jj":0.91,"coverage":0.67,"margin_ratio":0.21}
  }
}
```

> 若你希望更贴合“NPC 指令”，可以在响应中增加 `action` 字段（例如 `{"action":"FIND_ITEM","target_item_id":"veh-01"}`），但保持可选，避免把对话/行为系统耦合进检索服务。

---

## 7. 性能与工程优化（建议清单）

### 7.1 模型与向量计算

- 向量矩阵使用 `np.float32`，并在加载时做 L2 normalize，查询时只需点积（余弦相似）。
- 缓存 query embedding（短 TTL/LRU），应对高频相同 query。
- 对热路径避免 Python 循环：候选精排尽量向量化。

可选加速（按需开启）：

- ONNX Runtime 推理（CPU 高并发更稳）
- ANN：HNSW（hnswlib）或 Faiss（更快但依赖更重）

### 7.2 并发与一致性

- 索引对象只读共享；重建索引采用“构建新对象→原子切换引用”的方式，查询无锁或读锁极短。
- 管理写操作更新存储后，触发增量刷新；避免每次写都全量 rebuild。

### 7.3 典型量级建议

- < 5 万 items：exact 向量检索（矩阵乘法）+ BM25 足够
- 5 万 ~ 50 万 items：建议 ANN 或候选约束（candidate_ids）成为默认路径
- > 50 万 items：建议拆分索引（按地图/场景/类型分片）或升级到 ANN + 分片

---

## 8. 配置与可扩展点

建议统一配置入口（环境变量 + `config.json`）：

- `MODEL_NAME`：默认 `BAAI/bge-small-zh-v1.5`
- `DEVICE`：`cpu|cuda`
- `STORAGE_BACKEND`：`file|sqlite`
- `INDEX_BACKEND`：`exact|hnsw|faiss`（可选）
- `POS_BACKEND_DEFAULT`：`jieba|hanlp`
- `THRESHOLDS.*`：accept/clarify/reject、min_coverage、tau_margin

可扩展点（对应当前项目结构）：

- `item_search/pos.py`：新增 POS 后端
- `item_search/normalize.py`：扩展同义词/归一化规则
- `item_search/engine.py`：替换/扩展召回与精排策略（如加入类型先验、属性匹配）

---

## 9. 可观测性与调试

- 日志：记录 query、决策、耗时、候选规模（注意脱敏）
- 指标（可选）：QPS、P95 延迟、索引构建耗时、模型推理耗时
- `debug: true`：返回解析与评分分解（当前项目已支持，建议保留）

---

## 10. 安全与权限（最小但够用）

本服务常用于内网/本地；仍建议提供：

- 简单 API Key（HTTP Header）用于管理接口（写操作）
- 仅本机绑定（默认 `127.0.0.1`），或允许配置白名单网段
- 导入接口限制文件大小与字段校验，避免异常数据导致 OOM

---

## 11. 开发里程碑（可直接落地到迭代）

### M1：收敛 API（1~2 天）

- 明确只保留：Item 管理、Search API、Index 管理、Health
- 整理数据结构与导入导出格式（与现有 `items_sample.json` 兼容或提供迁移脚本）

### M2：持久化与增量索引（2~4 天）

- 实现 `file|sqlite` 存储后端抽象
- 支持 upsert + version/updated_at
- 索引：全量 rebuild + 增量 refresh（按 version/updated_at）

### M3：性能与质量（2~5 天）

- 引入候选约束（candidate_ids）为默认推荐用法
- 压测与优化：query embedding 缓存、批量矩阵计算、锁粒度
- 校准阈值与负样本分布的离线工具（读取日志/标注集）

### M4：可选增强（按需）

- ANN（hnsw/faiss）与分片索引
- 多语言/多模型并存（按场景加载）
- 管理 UI（可选，当前已有 demo 可扩展）

---

## 12. 风险与对策

- **冷启动下载模型慢**：提供预下载脚本/镜像；启动时 warmup；可选本地模型目录。
- **中文短 query 分词不稳**：保持现有 heuristics；必要时为常见短词建立词典与别名表。
- **数据频繁变更导致索引抖动**：采用增量刷新 + 批量合并更新；管理端提供“批量导入→一次 rebuild”模式。
- **误匹配导致 NPC 行为错误**：坚持 `ACCEPT/CLARIFY/REJECT` 三态，默认更保守；强制候选约束（附近/背包）。

---

## 13. 推荐的落地形态（最少依赖的“最终形态”）

- 一个 FastAPI 进程：
  - 提供 `/v1/items/*`、`/v1/index/*`、`/v1/item_search`
  - 存储：SQLite 单文件（WAL），无外部依赖
  - 索引：内存 + 原子切换；默认 exact，按需启用 ANN
  - 运行：Windows 开发、Linux 容器化部署（推荐）
