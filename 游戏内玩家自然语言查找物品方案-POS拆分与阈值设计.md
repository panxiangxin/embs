# 游戏内玩家自然语言查找物品：POS（JJ/NN）拆分检索 + 可校准“未匹配阈值”方案

## 1. 目标与约束

**目标**

1) 玩家输入自然语言（可含口语/动作/指代）时，稳定找到正确物品（Top1/TopK）。  
2) 当**确实没有合适物品**时，能稳定输出“未匹配/需要澄清”，且阈值可控、可解释、可校准。  
3) 结果可解释（告诉玩家/日志：名词匹配了什么、形容词匹配了什么、置信度为何高/低）。

**关键思想（你给的 JJ/NN 拆分）**

- 形容词（JJ）→ **属性/描述类**：颜色、材质、破损程度、稀有度、大小、形状等，用于匹配 item 的描述类 label。  
- 名词（NN）→ **对象/类型/名称**：卡车、钥匙、药水、门、箱子等，用于匹配 item 的名称/别名/类型名。  

示例：`红色的破烂东西` → `JJ: 红色, 破烂`；`NN: 东西`  
其中 `东西/物品/玩意/道具` 这类 **泛化名词**信息量很低，需特殊处理（见 4.4）。

---

## 2. 数据建模（决定上限）

对每个物品（item）建议至少维护这些字段：

```json
{
  "id": "item_001",
  "name": "铁钥匙",
  "aliases": ["钥匙", "铁制钥匙", "旧钥匙"],
  "type": "钥匙",
  "desc_labels": ["铁制", "金属", "旧", "生锈", "小"],
  "extra": { "rarity": "common", "color": null }
}
```

字段含义：

- `name/aliases/type`：用于 **NN 匹配**（名称/别名/类型词）。  
- `desc_labels`：用于 **JJ 匹配**（属性标签）。  

可选增强：

- `desc_text`：把 labels 组织成一段自然语言描述（用于 rerank / 兜底）。  
- 标签分组：颜色/材质/状态/形状/大小…（便于权重与冲突处理）。  

---

## 3. 离线准备（索引与标定）

### 3.1 统一归一化（必须做）

对 query 与所有 item 文本字段做同一套 normalize：

- 全角半角、大小写、标点空格、繁简（如有）、数字单位（“1个/一个/一枚”）。
- 同义词/别名归一：`卡车=货车=truck`，`破烂=破损=损坏`，`铁制=金属`…

> 同义词词典建议可热更新：`synonyms.json`（便于迭代，不绑死在代码里）。

### 3.2 POS 标注与短语结构

实现上可分两档：

- **轻量**：分词 + 词性（如 `jieba.posseg`），易部署，效果够用但有误差。  
- **高质量**：分词 + POS + 依存句法（如 HanLP/LTP 等），能更准地识别“哪个形容词修饰哪个名词”。  

你提的 JJ/NN 体系来自 Penn Tagset；中文实现时可映射：

- `JJ` ≈ `a/ad/an`（形容词及其修饰形式）  
- `NN` ≈ `n/nr/ns/nt/nz/vn`（名词/专名/名物化动词等）

### 3.3 建两套索引（推荐：召回更稳）

1) **Name/Type 索引（NN 用）**

- 倒排（BM25）用于：专有词、短词、拼写变体、含编号（“钥匙A-12”）。  
- 向量索引用于：语义同义（“药水/药剂/恢复瓶”）。

2) **Desc label 索引（JJ 用）**

- label 级向量索引：每个 `desc_label` 一个向量（或 label+同义扩展）。  
- 统计 label 的 `idf`：用于抑制“红色/大/好用”这种高频属性的影响。

### 3.4 背景分布（用于“未匹配阈值”）

为“阈值可控”准备两类**负样本分布**（核心）：

- `F_neg_name(s)`：在“无关名词 vs 任意 item(name/type)”情况下，向量相似度的经验 CDF。  
- `F_neg_desc(s)`：在“无关形容词 vs 任意 desc_label”情况下，相似度的经验 CDF。  

构建方式（离线一次即可，定期刷新）：

1) 随机采样若干“无关 token/短语”（或从玩家语料里抽常见词，再打乱配对）。  
2) 与随机 name/type 或 desc_label 计算相似度，得到大量负样本相似度集合。  
3) 排序后形成经验分布（也可拟合成 Beta/高斯混合，但经验分布最稳）。  

> 有了 `F_neg`，就能把“相似度阈值”变成“误匹配概率阈值”（更可控、更能跨场景复用）。

---

## 4. 在线检索流程（完整闭环）

### 4.1 Query → 结构化（POS 拆分）

输入：玩家原句 `utterance`

1) **意图过滤**：只要是“找物品/查背包/指向物体”类再进入检索；否则走其他系统。  
2) **清洗**：去动作词/礼貌词/指代词（可保留空间词给世界检索模块）。  
3) **分词 + POS** 得到 tokens：`[(词, POS), ...]`  
4) **抽取候选：**

- `NN_phrases`：所有名词短语（优先取依存句法 head noun；无句法时取“最后一个非泛化名词”作为 head）。  
- `JJ_phrases`：所有形容词短语（以及“形容词+程度副词”的组合，如“很破/特别大”）。  
- `Other`：数量、方位、否定（“不要红色”）、比较级（“更大”）等。

输出结构：

```json
{
  "nn": ["卡车"],
  "jj": ["红色", "破损"],
  "constraints": { "count": 1, "near": "我", "neg": [] }
}
```

### 4.2 候选召回（先全后准）

召回目标：得到一个候选集合 `Cand`（通常 50~500 个），保证“对的东西尽量在里面”。

1) **NN 召回（强约束）**

- BM25：`nn` 去搜 `name/aliases/type`，取 topN（如 50）。  
- 向量：`emb(nn)` 去搜 name/type 向量索引，取 topN（如 50）。  

合并得到 `Cand_nn`。

2) **JJ 召回（弱约束/补召回）**

对每个 `jj`：`emb(jj)` 搜 desc_label 向量索引取 topM label，再把这些 label 关联到 item 得到 `Cand_jj`。

3) **合并候选**

- 如果 `Cand_nn` 非空：`Cand = Cand_nn ∪ Cand_jj`  
- 如果 `Cand_nn` 为空但存在“有效名词”（非泛化名词）：可以加大 NN topN 或开启同义扩展后再召回一次。  
- 如果只有 `jj` 或名词全是泛化名词：`Cand = Cand_jj`，但后续要更严格（见 4.4）。

### 4.3 候选打分（NN 用来定“是什么”，JJ 用来定“像不像”）

对每个候选 item 计算：

#### 4.3.1 名词分数（Name Score）

对 `nn` 中每个名词短语 `n`：

- `s_bm25(n,item)`：BM25 归一化到 `[0,1]`（或 logit 映射）  
- `s_vec(n,item)`：`cos(emb(n), emb(name/type/alias))` 的最大值

聚合：

```
S_nn(item) = max_{n in NN}  max( w_bm25*s_bm25, w_vec*s_vec )
```

#### 4.3.2 形容词分数（Desc Score）

对每个形容词短语 `j`：

```
best(j,item) = max_{l in item.desc_labels} cos(emb(j), emb(l))
```

并做“覆盖度式打分”：

```
S_jj(item) = Σ_{j in JJ}  w(j) * clip(best(j,item), tau_cover)
coverage   = (# {j | best(j,item) >= tau_cover}) / max(|JJ|, 1)
```

其中：

- `clip(x, tau)`：`x < tau` 记 0，否则记 x（避免弱相关噪声累计）。  
- `w(j)`：建议用 `idf(j) * type_coef(j)`  
  - `idf`：高频属性（红色/大/新的）自动降权  
  - `type_coef`：颜色/材质/大小等可额外降权（避免“颜色劫持”）

#### 4.3.3 最终分数（Final）

```
S(item) = α * S_nn(item) + β * S_jj(item) + γ * S_full(item) - penalty(item)
```

说明：

- `S_full(item)`：可选兜底（用整句/整段描述的 embedding 去和 item 的 `desc_text` 比）  
- `penalty`：可选惩罚，例如“名词完全不匹配但靠颜色挤上来”的候选  

推荐起步权重（后续用数据调参）：

- 当 `NN` 有有效名词：`α=0.6~0.8, β=0.2~0.4, γ=0~0.1`  
- 当 `NN` 缺失/全泛化：`α=0~0.2, β=0.7~0.9, γ=0~0.2`（但阈值要更严格）

### 4.4 “未匹配阈值”与置信度（重点）

只用固定余弦阈值（比如 `0.35`）通常不稳：候选集大小、label 密度、模型变化都会影响分数分布。

这里给一个更稳、可解释、可校准的方法：**用背景分布把分数变成“误匹配概率”**。

#### 4.4.1 从相似度到“误匹配概率”（自动考虑候选规模）

假设某个分数 `x` 在“无关匹配”情况下的经验分布为 `F_neg(x)`（越大越少见）。

当你在 `M` 次比较里取最大值（例如：一个名词去对比 M 个候选 name），在纯随机情况下：

```
p_spurious = P(max >= x) = 1 - (F_neg(x))^M
confidence = 1 - p_spurious
```

解释：

- 候选越多（M 越大），随机“碰巧很像”的概率越高 → 阈值会自动更严。  
- 你可以直接设定想要的误匹配率，例如 `p_spurious <= 0.01`（1% 误报）。  

#### 4.4.2 建议的多门控（比单阈值更稳）

对最终 Top1 候选（得分 `S1`，次高 `S2`）建议使用**三道门**：

1) **名词门（NN gate）**（当存在有效名词时必须过）

- 取 `S_nn_top1`（Top1 物品的名词分数）  
- 用 `F_neg_name` 算 `p_nn = 1 - F_neg_name(S_nn_top1)^M_name`  
- 规则：`p_nn <= p_nn_accept`（如 0.01），否则判为“未匹配/需澄清”

2) **覆盖门（JJ coverage gate）**（当 query 有 JJ 时必须过）

- `coverage >= min_coverage`（如 0.5 或 “至少命中 1 个关键 JJ”）

3) **间隔门（margin gate）**（防止 top1/top2 纠缠）

```
margin_ratio = (S1 - S2) / max(|S1|, eps)
margin_ratio >= tau_margin
```

当 margin 很小，即使 top1 分数不低，也容易是“多个都差不多像” → 更适合提示澄清。

#### 4.4.3 处理“泛化名词”（东西/物品/玩意）

`东西/物品/道具/玩意` 等泛化名词：

- **不作为 head noun**（不参与 NN gate 的通过判定）  
- 如果 query 只有泛化名词 + JJ（例如“红色的破烂东西”）：
  - 允许用 JJ 检索，但把策略切到“更严格”：  
    - 提高 `min_coverage`、提高 `tau_cover`  
    - 或把 `p_spurious` 门槛从 0.01 收紧到 0.001  
  - 以及更偏向交互：返回 Top3 并追问“你指的是钥匙/衣服/武器/材料哪一类？”

#### 4.4.4 一个可落地的决策表（推荐）

把结果分成三类：`ACCEPT / CLARIFY / REJECT`

- **ACCEPT**：  
  - 有有效名词：`p_nn <= 0.01` 且 `coverage >= 0.5`（若有 JJ）且 `margin_ratio >= 0.15`  
  - 无有效名词：`p_final <= 0.001` 且 `coverage >= 0.7` 且 `margin_ratio >= 0.2`

- **CLARIFY**（返回候选并提问）：  
  - `0.01 < p_nn <= 0.05` 或 `margin_ratio` 偏小或 `coverage` 边缘  

- **REJECT**（明确未匹配）：  
  - `p_nn > 0.05`（有名词）  
  - 或候选集合为空 / 分数整体很低（可用 `p_final` 判断）

`p_final` 的获得方式：

- 你可以对最终分数 `S` 也构建一个背景分布 `F_neg_final`（通过随机打乱 JJ/NN 与 items 配对采样得到），同样用  
  `p_final = 1 - F_neg_final(S1) ^ |Cand|`

---

## 5. 参数如何“更好地设定”（可重复校准流程）

你需要的不是“拍脑袋相似度阈值”，而是“可控误报率”的阈值。

推荐最小闭环（50~200 条标注就能起飞）：

1) 收集 query 样本：玩家真实输入 + 正确 item_id（没有就先人工造一批）。  
2) 记录在线日志：`nn/jj/coverage/S_nn/S_jj/S1/S2/Cand_size`。  
3) 离线构建 `F_neg_name / F_neg_desc / F_neg_final`。  
4) 设定目标误报率（FPR）：例如 **1% 误匹配** → `p_spurious <= 0.01`。  
5) 在标注集上扫描参数（网格搜索）：
   - `tau_cover`、`min_coverage`、`tau_margin`、`topN/topM`、权重 `α/β`  
6) 输出一份“阈值配置”（可分场景：背包检索/场景检索/交易所检索）。

> 关键收益：当候选规模、label 密度、模型换代时，只要重建 `F_neg_*`，阈值体系仍然可用。

---

## 6. 结果返回（玩家体验 + 可解释）

建议返回结构：

```json
{
  "status": "ACCEPT|CLARIFY|REJECT",
  "best": {"id":"item_001","name":"铁钥匙"},
  "alternatives": [{"id":"item_002","name":"铜钥匙"}],
  "confidence": 0.97,
  "explain": {
    "nn": [{"phrase":"钥匙","match":"铁钥匙","score":0.62}],
    "jj": [{"phrase":"破烂","match":"旧","score":0.58},{"phrase":"红色","match":"锈色","score":0.41}],
    "coverage": 0.5,
    "margin_ratio": 0.18
  }
}
```

前端表现：

- ` slicing0 - ACCEPT：直接定位/高亮/选中物品  
- `CLARIFY`：展示 Top3 并问 1~2 个最关键的澄清问题（优先问名词类别）。  
- `REJECT`：明确告知“没找到”，给出可选引导（“你想找哪一类？武器/材料/钥匙…”）。  

---

## 7. 你给的例子如何落地

`红色的破烂东西`

- POS：`JJ=红色, 破烂`；`NN=东西(泛化)`  
- 处理：NN 视为无效 head → 走“JJ-only 严格模式”
  - 候选：由 `红色`/`破烂` 分别召回的 desc_labels 合并  
  - 打分：以 `coverage` 与 `p_final` 为主  
  - 若 Top3 很接近（margin 小）：CLARIFY  
  - 若整体 `p_final` 不够小：REJECT + 提问“你指的是哪一类东西？”

---

## 8. 实施建议（工程落点）

推荐服务端提供一个统一接口（客户端只传 query + 上下文候选集）：

- `POST /item_search`
  - 入参：`query`, `context_items`（背包/附近物品列表/全库）、`topk`、`threshold_profile`
  - 出参：`status + candidates + confidence + explain`

模块划分：

1) `normalize`（词典+规则）  
2) `pos_tag`（分词+词性+可选句法）  
3) `recall`（BM25 + 向量索引）  
4) `rank`（NN/JJ 组合打分）  
5) `confidence`（p_spurious + margin + coverage）  
6) `dialog`（澄清策略）  

---

## 9. 下一步我需要你补充的信息（可选，用于把方案参数化/落代码）

1) 物品库规模：背包几十个？场景几百个？全局几万？  
2) 你们现有 `name/labels` 字段实际长什么样（能否区分 type 与 desc_labels）？  
3) 10~20 条真实 query + 期望命中 item_id（越真实越好）。  
4) 你们更在意：误匹配（false positive）还是漏匹配（false negative）？

