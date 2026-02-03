## 0) 数据结构（和论文一致）

每个实体（entity/asset）至少要有：

* `entity_id`
* `labels: List[str]`（例如 `[broken, blue, car, vehicle, …]`）
  （location/rotation/bbox/cover points 你可以先留着不用于排序）

---

## 1) 离线预处理：把“标签集合”做成可检索索引

### 1.1 标签规范化（必须做）

对每个 label 做统一化，否则 embedding 会很飘：

* 全角半角、大小写、去多余空格/标点
* 同义词归一（truck/lorry/货车/卡车；house/building/房子/建筑物）
* 可选：把“红色卡车/货运”这种复合概念拆成 `{红色, 卡车, 货运}`（你现在已经这么做了，挺好）

### 1.2 建两套向量（推荐）

**A. label 级索引（用于召回）**
对每个 `(entity_id, label)` 计算 `emb(label)`，建 ANN 索引（FAISS/Milvus/pgvector 都行）。

**B. entity 级向量（用于补充打分/加速）**
把一个实体的 labels 聚合成一个向量：

* `emb_entity = normalize( Σ w(label) * emb(label) )`

> 这样做的目的是：别让 “红色” 这种通用词把结果“劫持”；后面我们会给它更低权重。

### 1.3 计算标签权重 w(label)（用 IDF 自动抑制“红色”）

统计每个 label 在多少实体里出现：`df(label)`，实体总数 `N`

* `idf(label) = log((N+1)/(df(label)+1)) + 1`
* `w(label) = idf(label)`（也可再乘一个类型系数，比如颜色类 *0.2）

这一步是你 case 的关键：**“红色”在很多实体都出现 → 权重自然变低**。

---

## 2) 在线检索 Ftext：从 query → 候选实体集合（top-k）

输入：上游（BERT-FID/LLM）给你的目标描述 `target_text`（论文也就是 “entity description input”）

### 2.1 Query 解析成短语集合 Q（别只用整句）

把 `target_text` 拆成若干关键词/短语（你示例里就是：红色、卡车、货运、车辆、重型…）
设 `Q = {q1, q2, …}`，并给每个 `q` 一个权重 `wq`（建议直接用 **idf(q)** 或 “名词高、属性低” 的规则）。

### 2.2 召回候选（candidate recall）

用 **label 级索引**做召回：

* 对每个 `q ∈ Q`：搜索 topM 个最相似 label（比如 M=50）
* 把返回的 label 所属的 `entity_id` 汇总成候选集合 `Cand`

> 这一步对应论文“对场景内所有 entity labels embeddings 做 similarity search”。

---

## 3) Entity 打分（核心改进点）：从“单标签 max”变成“多短语覆盖 + 权重”

你现在的问题根源是：`score(entity)=max_label_sim(query,label)`
改成下面这个（覆盖度聚合）：

### 3.1 覆盖度聚合分数（强烈推荐）

对每个候选实体 e：
[
score(e)=\sum_{q \in Q} w_q \cdot \max_{l \in Labels(e)} sim(emb(q), emb(l))
]

* `sim` 用 cosine
* `w_q` 用 idf 或规则（颜色低、名词高）
* 可加一个**阈值裁剪**：如果 `max sim < τ_cover` 就当作 0（表示“这个 q 没被覆盖到”）

这样：

* “红色卡车/货运”会被覆盖到多个 q（卡车/货运/车辆/重型…）
* “红色房子”基本只能覆盖到 红色、房子/建筑物
  结果自然拉开。

### 3.2 核心名词硬门控（推荐但不算“任务加权”）

从 Q 里挑一个“核心名词”（卡车/房子/门/箱子…），记为 `head`
如果：
[
\max_{l \in Labels(e)} sim(emb(head), emb(l)) < \tau_{head}
]
就直接把该实体淘汰（或 score 乘 0.1）。

> 这能从机制上避免“红色 + 任意物体”打平。

---

## 4) 置信度 Ctext（对应论文 Algorithm 2）

论文要求在 Ftext 之后“Compute the confidence Ctext”，低于阈值就走 CLIP。
你可以用一个很简单、可调参的定义：

* 先算所有候选实体的 `score(e)`，取前两名 `s1>=s2`
* 定义：

  * `Ctext = (s1 - s2) / max(|s1|, ε)`（margin ratio）
* 或者：`Ctext = softmax(scores)[0]`（top1 概率）

当出现你这种情况（只覆盖“红色”），往往 `s1≈s2` → **Ctext 很低**，就会触发 Fimage。

---

## 5) 可选：Fimage（fine-tuned CLIP）兜底（和论文一致）

当 `Ctext < δ2` 时，用 CLIP 做 text→image 检索 top-k 候选。
论文也明确：文本检索不足时用 fine-tuned CLIP 搜 image embeddings。
实现要点：

* 每个实体准备 1~N 张代表性视角图（或渲染图），离线算 `emb_img`
* 在线用 CLIP text encoder 算 `emb_text(target_text)`
* ANN 搜 topK entity

最后把两路结果合并（不涉及动态重排）：

* `final_score = α * normalize(score_text) + (1-α) * normalize(score_clip)`
* 或者：若触发 CLIP，就直接用 CLIP 排序返回

论文实验也显示“text similarity + fine-tuned CLIP”的组合优于单独方法。

---

## 6) 输出与可解释性（方便你调参）

每次返回 top-k 时，建议把这些一起打 log：

* 每个实体：`score(e)`
* 对每个 q：命中的最佳 label、相似度（让你看到是不是又被“红色”劫持）
* `Ctext`、是否触发 CLIP
* `head` 门控是否通过

---

## 7) 一份可直接照抄的伪代码

```python
def retrieve_entity_Ftext(target_text: str, scene_entities: List[Entity]) -> List[ScoredEntity]:
    Q = extract_phrases(target_text)              # ["红色","卡车","货运","车辆","重型",...]
    wq = {q: idf(q) * type_coef(q) for q in Q}    # 颜色类 type_coef=0.2, 名词=1.0 等

    # 1) 候选召回：label级索引
    Cand = set()
    for q in Q:
        for (label_id, sim) in label_index.search(emb(q), topM=50):
            Cand.add(label_id_to_entity[label_id])

    # 2) 核心名词门控
    head = pick_head_noun(Q)
    scored = []
    for e in Cand:
        head_sim = max(cos(emb(head), emb(l)) for l in e.labels)
        if head_sim < TAU_HEAD:
            continue

        # 3) 覆盖度聚合打分
        s = 0.0
        for q in Q:
            best = max(cos(emb(q), emb(l)) for l in e.labels)
            if best < TAU_COVER:
                best = 0.0
            s += wq[q] * best
        scored.append((e, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 4) 置信度
    s1 = scored[0][1] if scored else 0.0
    s2 = scored[1][1] if len(scored) > 1 else 0.0
    Ctext = (s1 - s2) / max(abs(s1), 1e-6)

    return scored, Ctext
```

---

## 用你这个例子会怎么变

* “红色卡车/货运”会在 Q 里有 **卡车/货运/车辆/重型** 多个强区分词
* “红色房子”最多覆盖 红色、房子/建筑物
* 又因为 `idf(红色)` 很低 / 颜色类系数很低 → 房子就算“红色”匹配很好，也很难追上卡车的多项累加

---

如果你把你们现在线上的三个参数告诉我（`topM / TAU_COVER / TAU_HEAD`），我可以给你一套**默认可用的参数起步值**（以及 Ctext 的 δ2 建议范围），让你直接试跑看“红色卡车 vs 红色房子”能不能稳定拉开。
