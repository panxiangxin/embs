from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import Heuristics, ParsedQuery, ParsedToken
from .normalize import Normalizer


@dataclass(frozen=True)
class PosVocab:
    known_nouns: set[str]
    known_desc: set[str]
    known_noun_suffixes: set[str]


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

_hanlp_pos = None
_hanlp_pipelines: dict[str, object] = {}


def _get_hanlp_pipeline(granularity: str | None) -> object:
    import hanlp

    global _hanlp_pos, _hanlp_pipelines
    key = (granularity or "").strip().lower()
    if key in ("fine", "细分"):
        key = "fine"
    elif key in ("coarse", "粗分", ""):
        key = "coarse"
    else:
        raise ValueError("granularity仅支持 coarse 或 fine")

    if _hanlp_pos is None:
        _hanlp_pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

    pipeline = _hanlp_pipelines.get(key)
    if pipeline is None:
        tok = hanlp.load(
            hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
            if key == "coarse"
            else hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH
        )
        pipeline = (
            hanlp.pipeline()
            .append(tok, output_key="tok")
            .append(_hanlp_pos, input_key="tok", output_key="pos")
        )
        _hanlp_pipelines[key] = pipeline

    return pipeline


def _analyze_with_hanlp(text: str, granularity: str | None = None) -> tuple[list[str], list[str]]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("text不能为空")
    doc = _get_hanlp_pipeline(granularity or "coarse")(raw)
    words = doc.get("tok") or []
    flags = doc.get("pos") or []
    return list(words), list(flags)


def _analyze_with_jieba(text: str) -> tuple[list[str], list[str]]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("text不能为空")
    from jieba import posseg

    words = list(posseg.cut(raw))
    return [w.word for w in words], [w.flag for w in words]


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


def parse_query(
    text: str,
    normalizer: Normalizer,
    vocab: PosVocab | None = None,
    pos_backend: Literal["hanlp", "jieba"] = "jieba",
    heuristics: Heuristics | None = None,
) -> ParsedQuery:
    raw = text or ""
    tokens: list[ParsedToken] = []
    nn: list[str] = []
    jj: list[str] = []

    heur = heuristics or Heuristics()
    head_whitelist = {normalizer.norm(h) for h in (heur.head_nouns or ()) if normalizer.norm(h)}
    short_color_chars = {c[0] for c in normalizer.color_labels if c}

    key = (pos_backend or "").strip().lower()
    if key == "jieba":
        words, flags = _analyze_with_jieba(raw)
    elif key == "hanlp" or not key:
        words, flags = _analyze_with_hanlp(raw, granularity="fine")
    else:
        raise ValueError("pos_backend仅支持 hanlp 或 jieba")
    for w, flag in zip(words, flags):
        t = normalizer.norm(w)
        if not t or t in STOP_TOKENS:
            continue

        # Heuristic: split "color+noun" compounds when noun is in vocab, e.g. "红车" -> ("红色","车").
        # This improves short queries and keeps latency low (no extra model calls).
        if vocab and heur.split_compounds and 2 <= len(t) <= max(2, int(heur.split_max_len)):
            split_done = False
            for c in short_color_chars:
                if not t.startswith(c):
                    continue
                tail = t[len(c) :]
                if not tail:
                    continue
                tail_n = normalizer.norm(tail)
                if tail_n in vocab.known_nouns or (tail_n in head_whitelist and tail_n in vocab.known_noun_suffixes):
                    jj.append(normalizer.norm(c))
                    nn.append(tail_n)
                    tokens.append(ParsedToken(text=normalizer.norm(c), pos="JJ(split)"))
                    tokens.append(ParsedToken(text=tail_n, pos="NN(split)"))
                    split_done = True
                    break
            if split_done:
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
