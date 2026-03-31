# Jina-CLIP-v2 本地部署服务

[Jina-CLIP-v2](https://huggingface.co/jinaai/jina-clip-v2) 是一个多语言多模态 Embedding 模型，支持文本和图像的统一向量表示。

## 特性

- 🌐 **多语言支持**: 支持 89 种语言的文本编码
- 🖼️ **多模态**: 同时支持文本和图像输入
- 📐 **Matryoshka 表示**: 可调整输出维度 (64-1024)，节省存储空间
- ⚡ **高性能**: 基于 Jina-XLM-RoBERTa + EVA02-L 架构
- 🔧 **OpenAI 兼容 API**: 兼容 OpenAI Embedding API 格式

## 模型规格

| 组件 | 规格 |
|------|------|
| 总参数量 | 0.9B (865M) |
| 文本编码器 | Jina-XLM-RoBERTa (561M) |
| 图像编码器 | EVA02-L14 (304M) |
| 最大文本长度 | 8,192 tokens |
| 图像分辨率 | 512×512 像素 |
| 输出维度 | 64-1024 (可配置) |

## 快速开始

### 1. 启动服务

#### Windows (PowerShell)
```powershell
.\start_jina_clip.ps1
```

#### 手动启动
```bash
# 安装依赖
pip install -r requirements-jina-clip.txt

# 启动服务
python jina_clip_service.py
```

服务将在 http://localhost:8001 启动。

> **注意**: 首次启动会自动下载模型（约 3.5GB）到本地 HuggingFace 缓存目录。

### 2. 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `jinaai/jina-clip-v2` | 模型名称 |
| `DEVICE` | `auto` | 运行设备 (auto/cuda/cpu/mps) |
| `TRUNCATE_DIM` | `1024` | 默认输出维度 (64-1024) |
| `DEFAULT_TASK` | `retrieval.passage` | 默认任务类型 |
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `8001` | 服务端口 |

### 3. API 使用示例

#### 文本 Embedding
```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["这是一段中文文本", "This is English"],
    "dimensions": 512
  }'
```

#### 图像 Embedding (URL)
```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"image": "https://example.com/image.jpg"}],
    "dimensions": 512
  }'
```

#### 图像 Embedding (Base64)
```bash
# 将图片转为 base64
IMAGE_BASE64=$(base64 -w 0 image.jpg)

curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": [{\"image\": \"data:image/jpeg;base64,${IMAGE_BASE64}\"}],
    \"dimensions\": 512
  }"
```

#### 混合模态 (文本+图像)
```bash
curl -X POST "http://localhost:8001/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"text": "查询文本"},
      {"image": "https://example.com/image.jpg"},
      {"text": "另一段文本"}
    ],
    "dimensions": 512
  }'
```

#### 计算相似度
```bash
curl -X POST "http://localhost:8001/v1/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "美丽的日落",
    "images": ["https://example.com/sunset.jpg"],
    "dimensions": 512
  }'
```

#### 重排序
```bash
curl -X POST "http://localhost:8001/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "美丽的日落",
    "documents": ["海滩日落", "城市建筑", "山间日出"],
    "top_n": 3
  }'
```

### 4. Python 客户端示例

```python
import requests

# 文本 embedding
response = requests.post("http://localhost:8001/v1/embeddings", json={
    "input": ["这是一段中文文本", "This is English"],
    "dimensions": 512
})
data = response.json()
print(f"Embedding 维度: {len(data['data'][0]['embedding'])}")

# 图像 + 文本混合
response = requests.post("http://localhost:8001/v1/embeddings", json={
    "input": [
        {"text": "描述图片的文本"},
        {"image": "https://example.com/image.jpg"}
    ]
})
```

### 5. 运行测试

```powershell
# 服务启动后，在另一个终端运行
python test_jina_clip.py
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/status` | GET | 服务状态 |
| `/v1/embeddings` | POST | 创建 embedding |
| `/v1/similarity` | POST | 计算相似度 |
| `/v1/rerank` | POST | 重排序 |

## 任务类型 (Task)

Jina-CLIP-v2 支持不同的任务类型前缀，用于优化特定场景的检索效果：

| 任务类型 | 用途 |
|----------|------|
| `retrieval.query` | 查询/问题 |
| `retrieval.passage` | 文档/段落 (默认) |
| `classification` | 分类任务 |
| `text-matching` | 文本匹配 |

## 性能优化建议

1. **CUDA 加速**: 确保安装了 PyTorch CUDA 版本
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **可选依赖**: 安装 FlashAttention 和 xFormers 可进一步加速
   ```bash
   pip install flash-attn xformers
   ```

3. **Matryoshka 维度**: 根据需求选择合适的输出维度
   - 存储敏感: 使用 256 或 512
   - 精度优先: 使用 1024 (完整维度)

## 模型缓存

模型会自动下载到 HuggingFace 缓存目录：
- Windows: `%USERPROFILE%\.cache\huggingface\hub`
- Linux/macOS: `~/.cache/huggingface/hub`

## 参考

- [HuggingFace 模型页](https://huggingface.co/jinaai/jina-clip-v2)
- [Jina AI 官方博客](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/)
- [技术报告](https://arxiv.org/abs/2412.08802)
