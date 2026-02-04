# AGENTS.md - 项目指南

本文档为 AI 编程助手提供项目背景、架构和开发指南。

## 项目概述

**embs** 是一个面向游戏场景的本地 Embedding 服务，提供基于自然语言的物品检索能力。

核心功能：
- `POST /v1/embeddings`：本地向量 API（默认模型 `BAAI/bge-small-zh-v1.5`）
- `POST /v1/items/load`：加载物品库并构建索引（BM25 + 向量 + 负样本分布）
- `POST /v1/item_search`：基于 POS（JJ/NN）拆分的物品检索，输出 `ACCEPT/CLARIFY/REJECT` 决策

## 技术栈

- **后端框架**：FastAPI + Uvicorn
- **向量模型**：sentence-transformers（默认 `BAAI/bge-small-zh-v1.5`）
- **分词/POS**：jieba（快速，默认）、HanLP（精确，可选）
- **数值计算**：NumPy
- **深度学习**：PyTorch
- **前端**：原生 HTML/CSS/JavaScript（无框架依赖）

## 项目结构

```
embs/
├── app.py                      # FastAPI 主入口，提供 REST API
├── item_search/                # 核心检索引擎模块
│   ├── __init__.py             # 模块导出
│   ├── engine.py               # ItemSearchEngine 主引擎（~690行核心逻辑）
│   ├── models.py               # 数据模型（Item, SearchResult, Thresholds 等）
│   ├── pos.py                  # POS 标注与查询解析（jieba/HanLP 双后端）
│   ├── bm25.py                 # BM25 索引实现
│   ├── negdist.py              # 负样本分布（用于 p_spurious 校准）
│   └── normalize.py            # 文本归一化、同义词映射
├── demo/                       # Web 演示界面
│   ├── index.html              # 主页面
│   ├── app.js                  # 前端逻辑（~846行）
│   └── styles.css              # 样式
├── scripts/
│   └── smoke_test_item_search.py  # 本地自测脚本
├── data/
│   └── items_sample.json       # 示例物品库
├── start.ps1                   # 一键启动脚本（PowerShell）
├── start_demo.ps1              # 启动演示页面
└── requirements.txt            # Python 依赖

# 设计文档（中文）
├── 游戏内玩家自然语言查找物品方案-POS拆分与阈值设计.md
├── 自然语言-物品检索不准确-原因分析与优化方案.md
├── label embedding 做相似度搜索.md
└── label embedding 做相似度搜索 -优化方案.md
```

## 核心架构

### 1. POS（词性）拆分策略

系统将查询拆分为两类词：
- **NN（名词）**：匹配物品名称、类型、别名（如"卡车"、"钥匙"）
- **JJ（形容词）**：匹配描述标签（如"红色"、"破损"、"金属"）

支持两种 POS 后端：
- `jieba`：快速，适合低延迟场景（默认）
- `hanlp`：更精确，但速度较慢

### 2. 混合召回机制

- **BM25**：字面匹配，适合专有名词、缩写、编号
- **向量召回**：语义匹配，适合同义词（"货车≈卡车"）

### 3. 评分与决策

分数组成：
```
S_total = α * S_nn + β * S_jj
```

其中：
- `S_nn`：名词匹配分数（向量相似度 + BM25）
- `S_jj`：形容词覆盖分数（带阈值裁剪）

决策状态：
- `ACCEPT`：高置信度匹配，可直接使用
- `CLARIFY`：需要澄清，返回候选列表
- `REJECT`：未匹配或置信度过低

### 4. 可校准阈值（p_spurious）

系统使用负样本分布将相似度转换为"误匹配概率"：
```
p_spurious = 1 - CDF(x)^M
```

关键阈值参数：
- `accept_p_nn`：名词匹配接受阈值（默认 0.01）
- `clarify_p_nn`：名词匹配澄清阈值（默认 0.05）
- `accept_p_jj`：纯 JJ 场景接受阈值（默认 0.001，更严格）
- `min_coverage`：JJ 覆盖度要求（默认 0.5）
- `tau_margin`：Top1/Top2 间隔要求（默认 0.15）

## 启动与运行

### 启动服务

```powershell
# 一键启动（创建虚拟环境、安装依赖、启动服务）
.\start.ps1

# 健康检查
curl.exe http://127.0.0.1:8000/health
```

### 加载物品库

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/items/load" `
  -H "Content-Type: application/json" `
  --data-binary "@data/items_sample.json"
```

### 执行检索

```powershell
# 基础检索
curl.exe -X POST "http://127.0.0.1:8000/v1/item_search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"破损的蓝色卡车\",\"debug\":true}"

# 限定候选集合（如背包/附近物品）
curl.exe -X POST "http://127.0.0.1:8000/v1/item_search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"红色的破烂东西\",\"candidate_ids\":[\"veh-01\",\"str-01\"],\"debug\":true}"
```

### 本地测试

```powershell
python .\scripts\smoke_test_item_search.py
```

### 打开 Web 演示

```powershell
# 方式 A：由 API 服务托管（推荐）
Start-Process "http://127.0.0.1:8000/demo/"

# 方式 B：直接打开本地文件
Start-Process .\demo\index.html
```

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 嵌入模型名称 |
| `DEVICE` | `cpu` | 运行设备（cpu/cuda） |

## 代码规范

### Python

- 使用 Python 3.10+ 类型注解（`str | None`、`list[str]` 等）
- 数据模型使用 `@dataclass(frozen=True)` 确保不可变性
- 核心引擎使用 NumPy 进行向量化计算
- 字符串使用双引号（与项目现有风格保持一致）

### 关键模块依赖关系

```
app.py
  └─ item_search/
      ├─ engine.py  → 依赖：models, pos, bm25, negdist, normalize
      ├─ models.py  → 纯数据定义，无依赖
      ├─ pos.py     → 依赖：models, normalize
      ├─ bm25.py    → 独立模块
      ├─ negdist.py → 依赖：numpy
      └─ normalize.py → 独立模块
```

## 物品数据格式

```json
{
  "items": [
    {
      "id": "veh-01",
      "name": "蓝色卡车",
      "type": ["卡车", "车", "汽车"],
      "aliases": ["货车"],
      "desc_labels": ["蓝色", "破损", "重型"]
    }
  ],
  "enable_bm25": true
}
```

字段说明：
- `id`：唯一标识
- `name`：显示名称
- `type`：类型（支持字符串或字符串数组）
- `aliases`：别名列表
- `desc_labels`：描述标签（用于 JJ 匹配）

## 调试与日志

开启 `debug: true` 时，API 返回包含以下解释信息：
- `parsed`：解析后的 NN/JJ/head_noun/tokens
- `explain`：详细评分分解（s_nn, s_jj, coverage, p_nn, p_jj, margin_ratio 等）

## 性能考虑

- 首次启动会下载模型（约 100MB+）
- POS 词典在 `load_items` 时预构建，避免每次查询重建
- 嵌入向量缓存于 `_Embedder` 类中
- 使用 NumPy 进行批量矩阵运算

## 扩展建议

如需修改或增强：

1. **添加新的 POS 后端**：在 `pos.py` 中实现 `_analyze_with_xxx` 函数，并在 `parse_query` 中注册
2. **调整评分权重**：修改 `SearchConfig.weights` 中的 `alpha_nn`, `beta_jj`, `gamma_bm25`
3. **扩展同义词**：在 `normalize.py` 的 `DEFAULT_SYNONYMS` 中添加
4. **自定义阈值策略**：修改 `Thresholds` 数据类及 `engine.py` 中的决策逻辑

## 参考文档

项目包含详细的中文设计文档：
- `游戏内玩家自然语言查找物品方案-POS拆分与阈值设计.md`：完整架构设计
- `自然语言-物品检索不准确-原因分析与优化方案.md`：问题诊断与优化
- `label embedding 做相似度搜索.md`：基础检索原理
- `label embedding 做相似度搜索 -优化方案.md`：改进方案
