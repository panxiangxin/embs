下面把论文里“**label embedding 做相似度搜索**（text similarity retrieval）”这一段的流程，按实现视角拆开讲清楚——它对应论文的 **Scene Recognition / Multi-modal Dynamic Entity Retrieval** 模块。

---

## 1) 先把游戏里每个可检索实体做成“资产（asset）+ label 集合”

论文把 3D 游戏资产结构化成一条条记录（entity / asset），每条至少包含：

* **Labels**：一组字符串标签（例如 *broken, blue, car, vehicle,…*，可理解为同义词/属性词集合）
* 位置信息 **Location [x,y,z]**、朝向 **Rotation**、包围盒 **Bounding Box**
* 以及战术相关的 **Cover Points** 等信息 

这些 **Labels** 就是后续做“label embedding”的文本语料来源：你可以把每个实体的 label 当作该实体的文本描述（可能是多词/多短语）。

---

## 2) 离线：为“所有 entity 的 labels”预先算好向量（label embeddings）

论文在运行时会“对场景里所有 entity labels 的 embedding”做相似度检索——这意味着它默认你已经把每个 label（或 label 拼接后的文本）编码成向量并建好索引。论文描述为：

> 对玩家给出的 entity description 输入，先对 **scene 中所有 entity labels 的 embeddings 做 similarity search** 

实现上常见有两种（论文没限定你必须哪一种，但逻辑一致）：

* **label 级**：每个 label 单独一个向量（entity 有多个向量）
* **entity 级**：把 labels 拼起来（或池化）变一个 entity 向量

检索时用余弦相似度/内积即可。

---

## 3) 在线：从玩家指令里抽“目标描述”，编码成 query embedding

整体管线是：先通过 BERT-FID / LLM 得到 intent 和 target，然后进入 entity retrieval。

在附录 Algorithm 2 里写得更明确：如果“has target”，就进入检索：先用 **Ftext** 从 asset dataset 里取 top-k。

这里的 **Ftext** 就对应“label embedding 相似度搜索”——也就是把 target（比如“red truck / the red door / that tree”一类短语）编码成 query embedding，然后去和 label embeddings 做 ANN / top-k 检索。

