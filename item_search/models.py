from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    aliases: tuple[str, ...] = ()
    type: str | None = None
    desc_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedToken:
    text: str
    pos: str


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    nn: tuple[str, ...] = ()
    jj: tuple[str, ...] = ()
    head_noun: str | None = None
    tokens: tuple[ParsedToken, ...] = ()


@dataclass(frozen=True)
class LoadItemsResult:
    item_count: int
    name_phrase_count: int
    desc_label_count: int
    neg_name_samples: int
    neg_desc_samples: int


@dataclass(frozen=True)
class Heuristics:
    # Compound split: "红车" -> ("红色", "车") to avoid losing attributes in short queries.
    split_compounds: bool = True
    split_max_len: int = 3
    head_nouns: tuple[str, ...] = ("车", "门", "箱", "塔")

    # NN suffix lexical boost: "车" should strongly match "卡车/货车/汽车" even if item.type doesn't include "车".
    nn_suffix_match: bool = True
    nn_suffix_boost_to: float = 0.95


@dataclass(frozen=True)
class Thresholds:
    accept_score: float = 0.55
    clarify_score: float = 0.45

    accept_p_nn: float = 0.01
    clarify_p_nn: float = 0.05
    accept_p_jj: float = 0.001
    clarify_p_jj: float = 0.01
    tau_cover: float = 0.35
    min_coverage: float = 0.5
    min_coverage_no_nn: float = 0.7
    tau_margin: float = 0.15
    tau_margin_no_nn: float = 0.2


@dataclass(frozen=True)
class Weights:
    alpha_nn: float = 0.7
    beta_jj: float = 0.3
    gamma_bm25: float = 0.1


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    recall_topn_bm25: int = 50
    recall_topn_name_vec: int = 50
    recall_topm_desc_label: int = 50
    thresholds: Thresholds = field(default_factory=Thresholds)
    weights: Weights = field(default_factory=Weights)
    heuristics: Heuristics = field(default_factory=Heuristics)
    enable_bm25: bool = True


SearchStatus = Literal["ACCEPT", "CLARIFY", "REJECT"]
PosBackend = Literal["hanlp", "jieba"]


@dataclass(frozen=True)
class MatchExplain:
    s_total: float
    s_nn: float
    s_nn_vec: float
    s_nn_bm25: float
    s_jj: float
    coverage: float
    matched_jj: tuple[dict[str, Any], ...]
    matched_nn: tuple[dict[str, Any], ...]
    margin_ratio: float
    p_nn: float | None
    p_jj: float | None


@dataclass(frozen=True)
class ScoredItem:
    id: str
    name: str
    score: float
    confidence: float
    explain: MatchExplain | None = None


@dataclass(frozen=True)
class SearchDecision:
    status: SearchStatus
    reason: str


@dataclass(frozen=True)
class SearchResult:
    decision: SearchDecision
    parsed: ParsedQuery
    best: ScoredItem | None
    alternatives: tuple[ScoredItem, ...] = ()


@dataclass(frozen=True)
class SearchRequest:
    query: str
    config: SearchConfig = field(default_factory=SearchConfig)
    debug: bool = False
    candidate_ids: tuple[str, ...] | None = None
    pos_backend: PosBackend = "jieba"
