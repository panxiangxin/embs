"""
Jina-CLIP-v2 本地部署服务
支持文本和图像的多模态 Embedding API
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from PIL import Image


# =============================================================================
# 配置
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "jinaai/jina-clip-v2")
DEVICE = os.getenv("DEVICE", "auto")
TRUNCATE_DIM = int(os.getenv("TRUNCATE_DIM", "1024"))  # Matryoshka 维度: 64-1024
DEFAULT_TASK = os.getenv("DEFAULT_TASK", "")  # Jina-CLIP-v2 只支持 retrieval.query 或不设置


# =============================================================================
# 全局模型实例
# =============================================================================

_model: Any = None
_device: str = "cpu"


def get_device() -> str:
    """自动检测或返回配置的设备"""
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return DEVICE


def load_model() -> Any:
    """加载 Jina-CLIP-v2 模型"""
    global _model, _device
    
    if _model is not None:
        return _model
    
    _device = get_device()
    print(f"[Jina-CLIP-v2] 正在加载模型: {MODEL_NAME}")
    print(f"[Jina-CLIP-v2] 使用设备: {_device}")
    print(f"[Jina-CLIP-v2] 默认 Matryoshka 维度: {TRUNCATE_DIM}")
    
    try:
        from transformers import AutoModel
        
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
        )
        _model = _model.to(_device)
        _model.eval()
        
        print(f"[Jina-CLIP-v2] 模型加载完成")
        return _model
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}") from e


def encode_text(texts: list[str], task: str | None = None, truncate_dim: int | None = None) -> list[list[float]]:
    """编码文本为 embedding 向量"""
    model = load_model()
    
    if not texts:
        return []
    
    dim = truncate_dim or TRUNCATE_DIM
    task_override = task or DEFAULT_TASK
    
    # 使用模型内置的 encode_text 方法
    # Jina-CLIP-v2 只支持 retrieval.query 或不设置 task
    with torch.no_grad():
        kwargs = {"truncate_dim": dim}
        if task_override and task_override in ["retrieval.query"]:
            kwargs["task"] = task_override
        embeddings = model.encode_text(texts, **kwargs)
        
        # 确保返回的是 numpy 数组或列表
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return list(embeddings)


def encode_image(
    images: list[Image.Image],
    truncate_dim: int | None = None,
) -> list[list[float]]:
    """编码图像为 embedding 向量"""
    model = load_model()
    
    if not images:
        return []
    
    dim = truncate_dim or TRUNCATE_DIM
    
    # 使用模型内置的 encode_image 方法
    with torch.no_grad():
        embeddings = model.encode_image(images, truncate_dim=dim)
        
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return list(embeddings)


def decode_image_input(image_input: str) -> Image.Image:
    """
    解码图像输入，支持:
    - URL (http/https)
    - Base64 编码的图片 (data:image/... 或直接 base64)
    - 本地文件路径
    """
    # 1. 检查是否为 URL
    if image_input.startswith(("http://", "https://")):
        import requests
        try:
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"无法下载图片 URL: {e}")
    
    # 2. 检查是否为 Data URI (base64)
    if image_input.startswith("data:image"):
        # 提取 base64 部分
        base64_data = image_input.split(",")[1]
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 3. 尝试直接解码 base64
    try:
        image_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        pass  # 不是 base64，继续尝试其他方式
    
    # 4. 检查是否为本地文件路径
    path = Path(image_input)
    if path.exists() and path.is_file():
        return Image.open(path).convert("RGB")
    
    raise ValueError(f"无法识别图片输入格式: {image_input[:50]}...")


# =============================================================================
# Pydantic 模型定义
# =============================================================================

class EmbeddingInput(BaseModel):
    """单个输入项，可以是文本或图像"""
    model_config = ConfigDict(extra="ignore")
    
    text: str | None = None
    image: str | None = None  # URL, base64, 或本地路径


class EmbeddingsRequest(BaseModel):
    """Embedding API 请求体"""
    model_config = ConfigDict(extra="ignore")
    
    model: str = MODEL_NAME
    input: list[str] | list[EmbeddingInput] | str | EmbeddingInput
    dimensions: int | None = Field(default=None, ge=64, le=1024)
    normalized: bool = True
    embedding_type: Literal["float", "binary", "ubinary"] = "float"
    task: str | None = None  # retrieval.query, retrieval.passage, etc.
    
    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int | None) -> int | None:
        if v is not None and v not in [64, 128, 256, 512, 768, 1024]:
            # Jina-CLIP-v2 支持任意 64-1024 之间的维度，但推荐标准值
            pass
        return v


class EmbeddingObject(BaseModel):
    """单个 embedding 结果"""
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    """Embedding API 响应体"""
    object: str = "list"
    model: str
    data: list[EmbeddingObject]
    usage: dict[str, int]


class SimilarityRequest(BaseModel):
    """相似度计算请求"""
    model_config = ConfigDict(extra="ignore")
    
    texts: list[str] | None = None
    images: list[str] | None = None  # URL/base64/路径列表
    query_text: str | None = None
    query_image: str | None = None
    dimensions: int | None = Field(default=None, ge=64, le=1024)


class SimilarityResult(BaseModel):
    """相似度计算结果"""
    text_to_image: list[list[float]] | None = None
    text_to_text: list[list[float]] | None = None
    image_to_image: list[list[float]] | None = None


# =============================================================================
# FastAPI 应用
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 启动时加载模型
    try:
        load_model()
    except Exception as e:
        print(f"[警告] 模型预加载失败: {e}")
    yield
    # 清理资源
    global _model
    _model = None


app = FastAPI(
    title="Jina-CLIP-v2 本地 Embedding 服务",
    description="支持多语言文本和图像的多模态 Embedding API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API 路由
# =============================================================================

@app.get("/health")
def health_check() -> dict[str, Any]:
    """健康检查"""
    return {
        "status": "ok" if _model is not None else "loading",
        "model": MODEL_NAME,
        "device": _device,
        "truncate_dim": TRUNCATE_DIM,
    }


@app.get("/v1/status")
def status() -> dict[str, Any]:
    """服务状态"""
    return {
        "model": MODEL_NAME,
        "device": _device,
        "truncate_dim": TRUNCATE_DIM,
        "default_task": DEFAULT_TASK,
        "ready": _model is not None,
    }


@app.post("/v1/embeddings")
def create_embeddings(payload: EmbeddingsRequest) -> EmbeddingsResponse:
    """
    创建文本或图像的 embedding
    
    示例请求:
    ```json
    {
        "input": ["一段文本", "另一段文本"],
        "dimensions": 512
    }
    ```
    
    或混合输入:
    ```json
    {
        "input": [
            {"text": "描述图片的文本"},
            {"image": "https://example.com/image.jpg"}
        ],
        "dimensions": 512
    }
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    # 处理输入
    texts: list[str] = []
    images: list[Image.Image] = []
    text_indices: list[int] = []
    image_indices: list[int] = []
    
    inputs = payload.input
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, EmbeddingInput):
        inputs = [inputs]
    
    for i, item in enumerate(inputs):
        if isinstance(item, str):
            # 自动检测是文本还是图片 URL/base64
            if item.startswith(("http://", "https://", "data:image")) or item.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                try:
                    img = decode_image_input(item)
                    images.append(img)
                    image_indices.append(i)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"第 {i} 项图片解析失败: {e}")
            else:
                texts.append(item)
                text_indices.append(i)
        elif isinstance(item, EmbeddingInput):
            if item.text is not None:
                texts.append(item.text)
                text_indices.append(i)
            elif item.image is not None:
                try:
                    img = decode_image_input(item.image)
                    images.append(img)
                    image_indices.append(i)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"第 {i} 项图片解析失败: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"第 {i} 项必须提供 text 或 image")
        else:
            raise HTTPException(status_code=400, detail=f"不支持的输入类型: {type(item)}")
    
    # 编码
    dim = payload.dimensions or TRUNCATE_DIM
    
    try:
        text_embeddings = encode_text(texts, task=payload.task, truncate_dim=dim) if texts else []
        image_embeddings = encode_image(images, truncate_dim=dim) if images else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"编码失败: {e}")
    
    # 合并结果
    total_len = len(inputs)
    results: list[EmbeddingObject | None] = [None] * total_len
    
    for idx, emb in zip(text_indices, text_embeddings):
        results[idx] = EmbeddingObject(index=idx, embedding=emb)
    for idx, emb in zip(image_indices, image_embeddings):
        results[idx] = EmbeddingObject(index=idx, embedding=emb)
    
    # 过滤 None（理论上不应该有）
    final_results = [r for r in results if r is not None]
    
    return EmbeddingsResponse(
        model=payload.model or MODEL_NAME,
        data=final_results,
        usage={
            "prompt_tokens": sum(len(t) for t in texts) if texts else 0,
            "total_tokens": sum(len(t) for t in texts) if texts else 0,
        },
    )


@app.post("/v1/similarity")
def compute_similarity(payload: SimilarityRequest) -> JSONResponse:
    """
    计算文本与图像之间的相似度
    
    支持:
    - 文本到图像相似度 (query_text + images)
    - 文本到文本相似度 (query_text + texts)
    - 图像到图像相似度 (query_image + images)
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    dim = payload.dimensions or TRUNCATE_DIM
    result: dict[str, Any] = {}
    
    try:
        # 文本查询
        if payload.query_text:
            query_emb = encode_text([payload.query_text], task="retrieval.query", truncate_dim=dim)[0]
            query_tensor = torch.tensor(query_emb)
            
            # 文本到图像
            if payload.images:
                pil_images = [decode_image_input(img) for img in payload.images]
                img_embs = encode_image(pil_images, truncate_dim=dim)
                img_tensor = torch.tensor(img_embs)
                similarities = (query_tensor @ img_tensor.T).tolist()
                result["text_to_image"] = [[s] for s in similarities]
            
            # 文本到文本
            if payload.texts:
                text_embs = encode_text(payload.texts, truncate_dim=dim)
                text_tensor = torch.tensor(text_embs)
                similarities = (query_tensor @ text_tensor.T).tolist()
                result["text_to_text"] = [[s] for s in similarities]
        
        # 图像查询
        if payload.query_image:
            query_img = decode_image_input(payload.query_image)
            query_emb = encode_image([query_img], truncate_dim=dim)[0]
            query_tensor = torch.tensor(query_emb)
            
            # 图像到图像
            if payload.images:
                pil_images = [decode_image_input(img) for img in payload.images]
                img_embs = encode_image(pil_images, truncate_dim=dim)
                img_tensor = torch.tensor(img_embs)
                similarities = (query_tensor @ img_tensor.T).tolist()
                result["image_to_image"] = [[s] for s in similarities]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算相似度失败: {e}")
    
    return JSONResponse(content=result)


class RerankRequest(BaseModel):
    """重排序请求体"""
    model_config = ConfigDict(extra="ignore")
    
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResult(BaseModel):
    """重排序结果项"""
    index: int
    document: str
    score: float


@app.post("/v1/rerank")
def rerank(payload: RerankRequest) -> JSONResponse:
    """
    对候选结果进行重排序（基于与查询的相似度）
    
    请求体:
    ```json
    {
        "query": "查询文本",
        "documents": ["文档1", "文档2", ...],
        "top_n": 5
    }
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    query = payload.query
    documents = payload.documents
    top_n = payload.top_n or len(documents)
    
    if not query or not documents:
        raise HTTPException(status_code=400, detail="必须提供 query 和 documents")
    
    try:
        # 编码查询和文档
        query_emb = torch.tensor(encode_text([query], task="retrieval.query", truncate_dim=TRUNCATE_DIM)[0])
        doc_embs = torch.tensor(encode_text(documents, truncate_dim=TRUNCATE_DIM))
        
        # 计算相似度
        similarities = (query_emb @ doc_embs.T).tolist()
        
        # 排序
        ranked = sorted(
            [{"index": i, "document": documents[i], "score": similarities[i]} for i in range(len(documents))],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_n]
        
        return JSONResponse({
            "results": ranked,
            "model": MODEL_NAME,
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重排序失败: {e}")


# 兼容 OpenAI API 格式的别名端点
@app.post("/v1/embeddings/encode")
def encode_endpoint(payload: EmbeddingsRequest) -> EmbeddingsResponse:
    """/v1/embeddings 的别名"""
    return create_embeddings(payload)


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Jina-CLIP-v2 本地 Embedding 服务                  ║
╠══════════════════════════════════════════════════════════════╣
║  模型: {MODEL_NAME:<50} ║
║  设备: {_device:<50} ║
║  默认维度: {TRUNCATE_DIM:<46} ║
║  服务地址: http://{host}:{port:<44} ║
╠══════════════════════════════════════════════════════════════╣
║  API 端点:                                                   ║
║    - POST /v1/embeddings    文本/图像 Embedding              ║
║    - POST /v1/similarity    相似度计算                       ║
║    - POST /v1/rerank        重排序                           ║
║    - GET  /health           健康检查                         ║
║    - GET  /v1/status        服务状态                         ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(app, host=host, port=port)
