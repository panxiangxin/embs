#!/usr/bin/env python3
"""
Jina-CLIP-v2 API 测试脚本
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

try:
    import requests
    from PIL import Image
except ImportError:
    print("请先安装依赖: pip install requests pillow")
    sys.exit(1)


BASE_URL = "http://127.0.0.1:8001"


def create_gradient_image() -> str:
    """创建一个简单的测试图片并返回 base64"""
    # 创建一个渐变色的测试图片
    img = Image.new('RGB', (256, 256), color='red')
    pixels = img.load()
    for i in range(256):
        for j in range(256):
            pixels[i, j] = (i, j, 128)
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def test_health():
    """测试健康检查"""
    print("\n[1] 测试健康检查...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_text_embedding():
    """测试文本 embedding"""
    print("\n[2] 测试文本 Embedding...")
    try:
        payload = {
            "input": ["这是一段中文文本", "This is an English text", "美しい日本語のテキスト"],
            "dimensions": 512
        }
        resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  返回 embedding 数量: {len(data['data'])}")
            print(f"  向量维度: {len(data['data'][0]['embedding'])}")
            return True
        else:
            print(f"  错误: {resp.text}")
            return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_image_embedding():
    """测试图像 embedding"""
    print("\n[3] 测试图像 Embedding...")
    try:
        # 创建一个 base64 图片
        img_base64 = create_gradient_image()
        
        payload = {
            "input": [{"image": img_base64}],
            "dimensions": 512
        }
        resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  返回 embedding 数量: {len(data['data'])}")
            print(f"  向量维度: {len(data['data'][0]['embedding'])}")
            return True
        else:
            print(f"  错误: {resp.text}")
            return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_multimodal_embedding():
    """测试混合模态 embedding"""
    print("\n[4] 测试混合模态 Embedding（文本+图像）...")
    try:
        img_base64 = create_gradient_image()
        
        payload = {
            "input": [
                {"text": "这是一张渐变色图片的描述"},
                {"image": img_base64},
                {"text": "另一个文本描述"}
            ],
            "dimensions": 512
        }
        resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  返回 embedding 数量: {len(data['data'])}")
            print(f"  向量维度: {len(data['data'][0]['embedding'])}")
            return True
        else:
            print(f"  错误: {resp.text}")
            return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_similarity():
    """测试相似度计算"""
    print("\n[5] 测试相似度计算...")
    try:
        img_base64 = create_gradient_image()
        
        payload = {
            "query_text": "渐变色图片",
            "images": [img_base64],
            "dimensions": 512
        }
        resp = requests.post(f"{BASE_URL}/v1/similarity", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  相似度结果: {data}")
            return True
        else:
            print(f"  错误: {resp.text}")
            return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def test_rerank():
    """测试重排序"""
    print("\n[6] 测试重排序...")
    try:
        payload = {
            "query": "美丽的日落",
            "documents": [
                "海滩上美丽的日落",
                "城市中的高楼大厦",
                "山上日出的美景",
                "海滩上的日落非常漂亮"
            ],
            "top_n": 3
        }
        resp = requests.post(f"{BASE_URL}/v1/rerank", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  重排序结果:")
            for i, result in enumerate(data.get("results", [])):
                print(f"    {i+1}. [{result['score']:.4f}] {result['document']}")
            return True
        else:
            print(f"  错误: {resp.text}")
            return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    print("=" * 50)
    print("Jina-CLIP-v2 API 测试")
    print("=" * 50)
    print(f"测试地址: {BASE_URL}")
    
    # 等待服务就绪
    print("\n等待服务就绪...")
    import time
    for i in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                break
        except:
            pass
        time.sleep(1)
        print(f"  等待中... ({i+1}s)", end="\r")
    else:
        print("\n服务未就绪，请检查服务是否已启动")
        sys.exit(1)
    
    print("\n服务已就绪，开始测试...")
    
    results = []
    results.append(("健康检查", test_health()))
    results.append(("文本 Embedding", test_text_embedding()))
    results.append(("图像 Embedding", test_image_embedding()))
    results.append(("混合模态 Embedding", test_multimodal_embedding()))
    results.append(("相似度计算", test_similarity()))
    results.append(("重排序", test_rerank()))
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\n总计: {passed_count}/{len(results)} 项通过")


if __name__ == "__main__":
    main()
