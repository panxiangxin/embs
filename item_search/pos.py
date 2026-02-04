from __future__ import annotations

from dataclasses import dataclass

from .models import ParsedQuery, ParsedToken
from .normalize import Normalizer


@dataclass(frozen=True)
class PosVocab:
    known_nouns: set[str]
    known_desc: set[str]

POS_API_URL = "http://127.0.0.1:32123/analyze"


STOP_TOKENS: set[str] = {
    "的",
    "了",
    "啊",
    "呀",
    "呢",
    "吧",
    "嘛",
    "么",
    "吗",
    "請",
    "请",
    "帮",
    "帮我",
    "我",
    "你",
    "一下",
    "一个",
    "一個",
    "这",
    "那",
    "这个",
    "那个",
    "这里",
    "那里",
    "这边",
    "那边",
    "附近",
}


def _is_noun_flag(flag: str) -> bool:
    if not flag:
        return False
    fu = flag.upper()
    if fu in {"NN", "NR", "NT", "NZ"}:
        return True
    if flag.startswith("n"):
        return True
    if flag in {"vn", "eng", "nz"}:
        return True
    return False


def _is_adj_flag(flag: str) -> bool:
    if not flag:
        return False
    fu = flag.upper()
    if fu in {"JJ", "VA"}:
        return True
    if flag.startswith("a"):
        return True
    if flag in {"b"}:
        return True
    return False


def parse_query(text: str, normalizer: Normalizer, vocab: PosVocab | None = None) -> ParsedQuery:
    import json
    import urllib.request

    raw = text or ""
    tokens: list[ParsedToken] = []
    nn: list[str] = []
    jj: list[str] = []

    req = urllib.request.Request(
        POS_API_URL,
        data=json.dumps({"text": raw,"granularity":"fine"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    words = payload.get("tokens") or []
    flags = payload.get("pos") or []
    for w, flag in zip(words, flags):
        t = normalizer.norm(w)
        if not t or t in STOP_TOKENS:
            continue
        flag = str(flag or "")
        tokens.append(ParsedToken(text=t, pos=flag))

        # Domain rule: colors are treated as JJ even if POS tagging is noisy
        if normalizer.type_coef(t) < 1.0:
            jj.append(t)
            continue

        if vocab and t in vocab.known_desc:
            jj.append(t)
            continue
        if vocab and t in vocab.known_nouns:
            nn.append(t)
            continue

        if _is_adj_flag(flag):
            jj.append(t)
        elif _is_noun_flag(flag):
            nn.append(t)

    nn_unique: list[str] = []
    seen = set()
    for n in nn:
        if n in seen:
            continue
        seen.add(n)
        nn_unique.append(n)

    jj_unique: list[str] = []
    seen = set()
    for j in jj:
        if j in seen:
            continue
        seen.add(j)
        jj_unique.append(j)

    head_noun = None
    for n in reversed(nn_unique):
        if normalizer.is_generic_noun(n):
            continue
        head_noun = n
        break

    return ParsedQuery(
        raw=raw,
        nn=tuple(nn_unique),
        jj=tuple(jj_unique),
        head_noun=head_noun,
        tokens=tuple(tokens),
    )
