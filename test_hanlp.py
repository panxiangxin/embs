# -*- coding: utf-8 -*-
from __future__ import annotations

import hanlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HanLP 分词与词性标注 API")


class AnalyzeRequest(BaseModel):
    text: str
    granularity: str | None = None


class AnalyzeResponse(BaseModel):
    tokens: list[str]
    pos: list[str]


_pos = None
_pipelines = {}


def _get_pipeline(granularity: str):
    global _pos, _pipelines
    key = (granularity or "").strip().lower()
    if key in ("fine", "细分"):
        key = "fine"
    elif key in ("coarse", "粗分", ""):
        key = "coarse"
    else:
        raise ValueError("granularity仅支持 coarse 或 fine")

    if _pos is None:
        _pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

    pipeline = _pipelines.get(key)
    if pipeline is None:
        tok = hanlp.load(
            hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
            if key == "coarse"
            else hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH
        )
        pipeline = (
            hanlp.pipeline()
            .append(tok, output_key="tok")
            .append(_pos, input_key="tok", output_key="pos")
        )
        _pipelines[key] = pipeline

    return pipeline


def _analyze(text: str, granularity: str | None = None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("text不能为空")
    doc = _get_pipeline(granularity or "coarse")(text)
    return {"tokens": doc["tok"], "pos": doc["pos"]}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_post(req: AnalyzeRequest):
    try:
        return _analyze(req.text, req.granularity)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze_get(text: str, granularity: str | None = None):
    try:
        return _analyze(text, granularity)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _analyze_jieba(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("text不能为空")
    from jieba import posseg
    words = list(posseg.cut(text))
    return {"tokens": [w.word for w in words], "pos": [w.flag for w in words]}


@app.post("/analyze_jieba", response_model=AnalyzeResponse)
def analyze_jieba_post(req: AnalyzeRequest):
    try:
        return _analyze_jieba(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analyze_jieba", response_model=AnalyzeResponse)
def analyze_jieba_get(text: str):
    try:
        return _analyze_jieba(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("test_hanlp:app", host="0.0.0.0", port=32123, reload=False)
