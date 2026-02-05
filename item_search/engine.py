from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .bm25 import BM25Index
from .models import (
    Heuristics,
    Item,
    LoadItemsResult,
    MatchExplain,
    ParsedQuery,
    ScoredItem,
    SearchConfig,
    SearchDecision,
    SearchRequest,
    SearchResult,
)
from .negdist import NegDistribution, build_neg_distribution
from .normalize import Normalizer
from .pos import PosVocab, parse_query


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


@dataclass(frozen=True)
class _IndexedItem:
    item: Item
    name_phrases: tuple[str, ...]
    type_phrase_count: int
    desc_labels: tuple[str, ...]


@dataclass(frozen=True)
class _Index:
    items: tuple[_IndexedItem, ...]
    id_to_pos: dict[str, int]
    bm25: BM25Index | None
    pos_vocab: PosVocab
    name_phrase_vectors: np.ndarray
    name_phrase_owner: np.ndarray  # phrase_idx -> item_pos
    item_name_vector: np.ndarray  # (item_count, dim)
    desc_label_vectors: np.ndarray
    desc_labels: tuple[str, ...]
    desc_label_to_idx: dict[str, int]
    desc_label_to_items: dict[str, tuple[int, ...]]  # label -> item_pos list
    desc_idf: dict[str, float]
    neg_name: NegDistribution
    neg_desc: NegDistribution


class _Embedder:
    def __init__(self, model) -> None:
        self._model = model
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        missing: list[str] = []
        order: list[str] = []
        for t in texts:
            key = t
            order.append(key)
            if key not in self._cache:
                missing.append(key)

        if missing:
            vecs = self._model.encode(missing, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype=np.float32)
            for i, t in enumerate(missing):
                self._cache[t] = vecs[i]

        stacked = np.stack([self._cache[t] for t in order], axis=0)
        return stacked.astype(np.float32, copy=False)


def _topk_indices(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    k = min(k, int(scores.shape[0]))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.tolist()


def _cos_sim_matrix(vecs: np.ndarray, q: np.ndarray) -> np.ndarray:
    # vecs and q are L2-normalized
    return vecs @ q


class ItemSearchEngine:
    def __init__(self, model, normalizer: Normalizer | None = None) -> None:
        self._normalizer = normalizer or Normalizer()
        self._embedder = _Embedder(model)
        self._index: _Index | None = None

    def loaded_item_count(self) -> int:
        return len(self._index.items) if self._index is not None else 0

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    def load_items(self, items: list[Item], enable_bm25: bool = True) -> LoadItemsResult:
        if not items:
            self._index = None
            return LoadItemsResult(
                item_count=0,
                name_phrase_count=0,
                desc_label_count=0,
                neg_name_samples=0,
                neg_desc_samples=0,
            )

        import re

        def _iter_type_phrases(raw_type: object) -> list[str]:
            if raw_type is None:
                return []
            if isinstance(raw_type, str):
                parts = re.split(r"[,\uFF0C\u3001/|;；\r\n\t]+", raw_type)
                return [p for p in (s.strip() for s in parts) if p]
            if isinstance(raw_type, (list, tuple, set)):
                out: list[str] = []
                for v in raw_type:
                    if v is None:
                        continue
                    if isinstance(v, str):
                        out.append(v)
                    else:
                        out.append(str(v))
                return out
            return [str(raw_type)]

        indexed: list[_IndexedItem] = []
        for it in items:
            name = self._normalizer.norm(it.name)
            aliases = [self._normalizer.norm(a) for a in it.aliases]
            type_phrases = _unique_preserve_order([self._normalizer.norm(t) for t in _iter_type_phrases(it.type)])
            desc = [self._normalizer.norm(l) for l in it.desc_labels]

            merged: list[str] = []
            seen: set[str] = set()
            type_phrase_count = 0

            if name:
                seen.add(name)
                merged.append(name)

            for t in type_phrases:
                if not t or t in seen:
                    continue
                seen.add(t)
                merged.append(t)
                type_phrase_count += 1

            for a in aliases:
                if not a or a in seen:
                    continue
                seen.add(a)
                merged.append(a)

            name_phrases = merged
            desc_labels = _unique_preserve_order(desc)
            indexed.append(
                _IndexedItem(
                    item=Item(
                        id=str(it.id),
                        name=it.name,
                        aliases=tuple(_unique_preserve_order(list(it.aliases))),
                        type=it.type,
                        desc_labels=tuple(_unique_preserve_order(list(it.desc_labels))),
                    ),
                    name_phrases=tuple(name_phrases),
                    type_phrase_count=int(type_phrase_count),
                    desc_labels=tuple(desc_labels),
                )
            )

        id_to_pos = {it.item.id: i for i, it in enumerate(indexed)}

        # BM25 docs (name/type/aliases)
        bm25 = None
        if enable_bm25:
            import jieba

            docs: list[list[str]] = []
            for it in indexed:
                doc_text = " ".join(it.name_phrases)
                docs.append([t for t in jieba.cut(doc_text) if t.strip()])
            bm25 = BM25Index.build(docs)

        # Name phrase vectors
        all_name_phrases: list[str] = []
        name_phrase_owner: list[int] = []
        for pos, it in enumerate(indexed):
            for p in it.name_phrases:
                all_name_phrases.append(p)
                name_phrase_owner.append(pos)

        name_phrase_vecs = self._embedder.embed(all_name_phrases) if all_name_phrases else np.zeros((0, 0), np.float32)
        name_phrase_owner_arr = np.asarray(name_phrase_owner, dtype=np.int32)

        # Item aggregated name vector
        if name_phrase_vecs.size:
            dim = int(name_phrase_vecs.shape[1])
            item_name = np.zeros((len(indexed), dim), dtype=np.float32)
            for phrase_idx, owner_pos in enumerate(name_phrase_owner_arr.tolist()):
                item_name[owner_pos] += name_phrase_vecs[phrase_idx]
            norms = np.linalg.norm(item_name, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            item_name = item_name / norms
        else:
            item_name = np.zeros((len(indexed), 0), dtype=np.float32)

        # Desc label vectors
        all_desc_labels = _unique_preserve_order([l for it in indexed for l in it.desc_labels])
        desc_label_vecs = self._embedder.embed(all_desc_labels) if all_desc_labels else np.zeros((0, 0), np.float32)
        desc_label_to_idx = {l: i for i, l in enumerate(all_desc_labels)}

        desc_label_to_items: dict[str, list[int]] = {}
        for pos, it in enumerate(indexed):
            for l in it.desc_labels:
                desc_label_to_items.setdefault(l, []).append(pos)
        desc_label_to_items_final = {k: tuple(v) for k, v in desc_label_to_items.items()}

        # desc idf
        df: dict[str, int] = {}
        for it in indexed:
            for l in set(it.desc_labels):
                df[l] = df.get(l, 0) + 1
        N = len(indexed) or 1
        desc_idf = {l: math.log((N + 1) / (c + 1)) + 1 for l, c in df.items()}

        # POS vocab (cached): avoid per-query rebuilding for low latency.
        import jieba

        known_desc: set[str] = set(all_desc_labels)
        for l in all_desc_labels:
            if l.endswith("制") and len(l) > 1:
                known_desc.add(l[:-1])
            for tok in jieba.cut(l):
                t = self._normalizer.norm(tok)
                if t:
                    known_desc.add(t)

        known_nouns: set[str] = set()
        for it in indexed:
            for p in it.name_phrases:
                if p:
                    known_nouns.add(p)
                for tok in jieba.cut(p):
                    t = self._normalizer.norm(tok)
                    if t:
                        known_nouns.add(t)

        known_noun_suffixes: set[str] = set()
        for n in known_nouns:
            if not n:
                continue
            if len(n) >= 1:
                known_noun_suffixes.add(n[-1:])
            if len(n) >= 2:
                known_noun_suffixes.add(n[-2:])

        pos_vocab = PosVocab(known_nouns=known_nouns, known_desc=known_desc, known_noun_suffixes=known_noun_suffixes)

        neg_name = build_neg_distribution(name_phrase_vecs)
        neg_desc = build_neg_distribution(desc_label_vecs)

        self._index = _Index(
            items=tuple(indexed),
            id_to_pos=id_to_pos,
            bm25=bm25,
            pos_vocab=pos_vocab,
            name_phrase_vectors=name_phrase_vecs,
            name_phrase_owner=name_phrase_owner_arr,
            item_name_vector=item_name,
            desc_label_vectors=desc_label_vecs,
            desc_labels=tuple(all_desc_labels),
            desc_label_to_idx=desc_label_to_idx,
            desc_label_to_items=desc_label_to_items_final,
            desc_idf=desc_idf,
            neg_name=neg_name,
            neg_desc=neg_desc,
        )

        return LoadItemsResult(
            item_count=len(indexed),
            name_phrase_count=len(all_name_phrases),
            desc_label_count=len(all_desc_labels),
            neg_name_samples=len(neg_name.samples),
            neg_desc_samples=len(neg_desc.samples),
        )

    def _ensure_index(self) -> _Index:
        if self._index is None:
            raise RuntimeError("No items loaded. Call load_items() or provide items via API.")
        return self._index

    def parse(self, query: str, pos_backend: str = "jieba", heuristics: Heuristics | None = None) -> ParsedQuery:
        idx = self._ensure_index()
        return parse_query(query, self._normalizer, vocab=idx.pos_vocab, pos_backend=pos_backend, heuristics=heuristics)

    def search(self, req: SearchRequest) -> SearchResult:
        cfg: SearchConfig = req.config
        idx = self._ensure_index()

        parsed = self.parse(req.query, pos_backend=req.pos_backend, heuristics=cfg.heuristics)
        valid_nns = [n for n in parsed.nn if not self._normalizer.is_generic_noun(n)]
        jjs = list(parsed.jj)
        if not valid_nns and not jjs:
            decision = SearchDecision(status="REJECT", reason="empty_query")
            return SearchResult(decision=decision, parsed=parsed, best=None, alternatives=())

        heur = cfg.heuristics
        head_whitelist = {self._normalizer.norm(h) for h in (heur.head_nouns or ()) if self._normalizer.norm(h)}

        # Effective weights: don't penalize missing channels
        has_nn = bool(valid_nns)
        has_jj = bool(jjs)
        if has_nn and has_jj:
            alpha = float(cfg.weights.alpha_nn)
            beta = float(cfg.weights.beta_jj)
            s = alpha + beta
            if s > 0:
                alpha_eff = alpha / s
                beta_eff = beta / s
            else:
                alpha_eff, beta_eff = 0.5, 0.5
        elif has_nn:
            alpha_eff, beta_eff = 1.0, 0.0
        else:
            alpha_eff, beta_eff = 0.0, 1.0

        allowed_pos: set[int] | None = None
        if req.candidate_ids is not None:
            allowed_pos = set()
            for cid in req.candidate_ids:
                p = idx.id_to_pos.get(cid)
                if p is not None:
                    allowed_pos.add(p)
            if not allowed_pos:
                decision = SearchDecision(status="REJECT", reason="candidate_ids_empty")
                return SearchResult(decision=decision, parsed=parsed, best=None, alternatives=())

        # Embeddings for query parts
        nn_vecs = self._embedder.embed(valid_nns) if valid_nns else np.zeros((0, 0), np.float32)
        jj_vecs = self._embedder.embed(jjs) if jjs else np.zeros((0, 0), np.float32)

        # Recall candidates
        cand_pos: set[int] = set(allowed_pos) if allowed_pos is not None else set()

        # NN recall: BM25
        bm25_scores: list[float] | None = None
        if cfg.enable_bm25 and idx.bm25 and valid_nns:
            import jieba

            q_tokens: list[str] = []
            for n in valid_nns:
                q_tokens.extend([t for t in jieba.cut(n) if t.strip()])
            bm25_scores = idx.bm25.score_all(q_tokens)
            if allowed_pos is None:
                top = BM25Index.topn(bm25_scores, cfg.recall_topn_bm25)
                for doc_idx, _ in top:
                    cand_pos.add(doc_idx)

        if allowed_pos is None:
            # NN recall: name vector (aggregated per item)
            if valid_nns and idx.item_name_vector.size:
                # Use max over nn phrases
                sims = []
                for qv in nn_vecs:
                    sims.append(_cos_sim_matrix(idx.item_name_vector, qv))
                sim_mat = np.stack(sims, axis=1) if sims else np.zeros((len(idx.items), 0), np.float32)
                nn_best = sim_mat.max(axis=1) if sim_mat.size else np.zeros((len(idx.items),), np.float32)
                top_items = _topk_indices(nn_best, cfg.recall_topn_name_vec)
                cand_pos.update(top_items)

            # JJ recall: label vector -> items
            if jjs and idx.desc_label_vectors.size:
                for qv in jj_vecs:
                    label_sims = _cos_sim_matrix(idx.desc_label_vectors, qv)
                    top_labels_idx = _topk_indices(label_sims, cfg.recall_topm_desc_label)
                    for li in top_labels_idx:
                        label = idx.desc_labels[li]
                        cand_pos.update(idx.desc_label_to_items.get(label, ()))

            if not cand_pos:
                cand_pos = set(range(len(idx.items)))

        cand_list = sorted(cand_pos)

        # Normalize BM25 inside candidates (0..1)
        bm25_norm_by_pos: dict[int, float] = {}
        if bm25_scores is not None and cand_list:
            max_s = max(bm25_scores[p] for p in cand_list) or 0.0
            if max_s > 0:
                for p in cand_list:
                    bm25_norm_by_pos[p] = float(bm25_scores[p] / max_s)

        # Score candidates
        scored: list[tuple[int, float, float, float, float, float, list[dict], list[dict]]] = []
        # tuple: (pos, total, s_nn, s_nn_vec, s_nn_bm25, s_jj, nn_matches, jj_matches)

        tau_cover = cfg.thresholds.tau_cover

        for p in cand_list:
            item = idx.items[p]

            s_nn_vec = 0.0
            nn_matches: list[dict] = []
            if valid_nns and idx.name_phrase_vectors.size:
                phrase_indices_all = np.where(idx.name_phrase_owner == p)[0]
                if phrase_indices_all.size:
                    phrase_indices = phrase_indices_all
                    base_offset = 0
                    if item.type_phrase_count > 0 and phrase_indices_all.size >= 2:
                        start = 1
                        end = min(int(1 + item.type_phrase_count), int(phrase_indices_all.size))
                        if end > start:
                            phrase_indices = phrase_indices_all[start:end]
                            base_offset = start

                    phrase_vecs = idx.name_phrase_vectors[phrase_indices]
                    phrase_texts = item.name_phrases[base_offset : base_offset + int(phrase_vecs.shape[0])]
                    best_over_nns = 0.0
                    best_detail = None
                    for qi, qv in enumerate(nn_vecs):
                        nn_text = valid_nns[qi]
                        sims = phrase_vecs @ qv
                        bi = int(np.argmax(sims))
                        sim = float(sims[bi])

                        # Lexical suffix match for head nouns: "车" should strongly match "卡车/货车/汽车".
                        # This improves short queries (e.g., "红车") even when type lists don't include the hypernym.
                        if heur.nn_suffix_match and nn_text in head_whitelist:
                            lex_idx: list[int] = []
                            for li, phrase in enumerate(phrase_texts):
                                if not phrase:
                                    continue
                                if phrase == nn_text or phrase.endswith(nn_text):
                                    lex_idx.append(li)
                            if lex_idx:
                                lex_idx_arr = np.asarray(lex_idx, dtype=np.int32)
                                lex_best = int(lex_idx_arr[int(np.argmax(sims[lex_idx_arr]))])
                                lex_sim = float(sims[lex_best])
                                boost_to = float(max(0.0, min(1.0, heur.nn_suffix_boost_to)))
                                lex_sim = max(lex_sim, boost_to)
                                if lex_sim > sim:
                                    bi = lex_best
                                    sim = lex_sim

                        if sim > best_over_nns:
                            best_over_nns = sim
                            local_idx = int(base_offset + bi)
                            best_detail = {
                                "nn": nn_text,
                                "name_phrase": item.name_phrases[local_idx] if local_idx < len(item.name_phrases) else None,
                                "sim": sim,
                            }
                    s_nn_vec = best_over_nns
                    if best_detail:
                        nn_matches.append(best_detail)

            s_nn_bm25 = bm25_norm_by_pos.get(p, 0.0)
            s_nn = max(s_nn_vec, cfg.weights.gamma_bm25 * s_nn_bm25)

            jj_weight_sum = 0.0
            jj_used_sum = 0.0
            jj_covered_weight_sum = 0.0
            jj_matches: list[dict] = []
            item_labels = [l for l in item.desc_labels if l in idx.desc_label_to_idx]
            label_vecs = (
                idx.desc_label_vectors[[idx.desc_label_to_idx[l] for l in item_labels]]
                if (jjs and idx.desc_label_vectors.size and item_labels)
                else np.zeros((0, 0), np.float32)
            )

            if jjs:
                for j_idx, j in enumerate(jjs):
                    qv = jj_vecs[j_idx]
                    w = float(idx.desc_idf.get(j, 1.0)) * float(self._normalizer.type_coef(j))
                    jj_weight_sum += w

                    sims = label_vecs @ qv if label_vecs.size else np.zeros((0,), np.float32)
                    if sims.size:
                        bi = int(np.argmax(sims))
                        sim = float(sims[bi])
                        used = float(sim if sim >= tau_cover else 0.0)
                        best_label = item_labels[bi] if bi < len(item_labels) else None
                    else:
                        bi = -1
                        sim = 0.0
                        used = 0.0
                        best_label = None

                    if used > 0:
                        jj_covered_weight_sum += w
                    jj_used_sum += w * used
                    jj_matches.append(
                        {
                            "jj": j,
                            "best_label": best_label,
                            "sim": float(sim),
                            "used": float(used),
                            "w": float(w),
                        }
                    )

            coverage = float(jj_covered_weight_sum / jj_weight_sum) if jj_weight_sum > 0 else 0.0
            s_jj = float(jj_used_sum / jj_weight_sum) if jj_weight_sum > 0 else 0.0

            total = alpha_eff * s_nn + beta_eff * s_jj

            scored.append((p, float(total), float(s_nn), float(s_nn_vec), float(s_nn_bm25), float(s_jj), nn_matches, jj_matches))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            decision = SearchDecision(status="REJECT", reason="empty_candidates")
            return SearchResult(decision=decision, parsed=parsed, best=None, alternatives=())

        top1 = scored[0]
        top2 = scored[1] if len(scored) > 1 else None
        s1 = top1[1]
        s2 = top2[1] if top2 else 0.0
        margin_ratio = (s1 - s2) / max(abs(s1), 1e-6)

        th = cfg.thresholds

        # Calibrated spurious-match probabilities (p_spurious) for gates.
        # - NN gate: use name/type phrase similarity (vector), corrected for multiple comparisons.
        # - JJ-only gate: multiply per-JJ spurious probabilities (strict by design).
        p_nn_top1: float | None = None
        if has_nn and idx.neg_name.samples:
            m_phrases = 0
            for p in cand_list:
                it = idx.items[p]
                if it.type_phrase_count > 0:
                    m_phrases += max(1, int(it.type_phrase_count))
                else:
                    m_phrases += max(1, len(it.name_phrases))
            m_name_comparisons = max(1, m_phrases * max(1, len(valid_nns)))
            p_nn_top1 = idx.neg_name.tail_p_max(float(top1[3]), m_name_comparisons)

        p_jj_top1: float | None = None
        if (not has_nn) and has_jj and idx.neg_desc.samples:
            p_pos = int(top1[0])
            m_labels = max(1, len(idx.items[p_pos].desc_labels))
            p_prod = 1.0
            any_used = False
            for m in top1[7]:
                if float(m.get("used") or 0.0) <= 0:
                    continue
                any_used = True
                sim = float(m.get("sim") or 0.0)
                p_prod *= max(1e-12, idx.neg_desc.tail_p_max(sim, m_labels))
            p_jj_top1 = p_prod if any_used else 1.0

        p_spurious_top1 = p_nn_top1 if p_nn_top1 is not None else p_jj_top1

        # Coverage from stored per-JJ weights.
        cov_weight_sum = float(sum(float(m.get("w") or 0.0) for m in top1[7])) if has_jj else 0.0
        cov_covered = float(sum(float(m.get("w") or 0.0) for m in top1[7] if float(m.get("used") or 0.0) > 0.0)) if has_jj else 0.0
        coverage_top1 = float(cov_covered / cov_weight_sum) if cov_weight_sum > 0 else 0.0

        # 1) Base decision from calibrated p_spurious when available; otherwise fall back to score thresholds.
        if has_nn and p_nn_top1 is not None:
            if p_nn_top1 <= th.accept_p_nn:
                decision = SearchDecision(status="ACCEPT", reason="p_nn_le_accept")
            elif p_nn_top1 <= th.clarify_p_nn:
                decision = SearchDecision(status="CLARIFY", reason="p_nn_le_clarify")
            else:
                decision = SearchDecision(status="CLARIFY", reason="p_nn_gt_clarify")
        elif (not has_nn) and p_jj_top1 is not None:
            if p_jj_top1 <= th.accept_p_jj:
                decision = SearchDecision(status="ACCEPT", reason="p_jj_le_accept")
            elif p_jj_top1 <= th.clarify_p_jj:
                decision = SearchDecision(status="CLARIFY", reason="p_jj_le_clarify")
            else:
                decision = SearchDecision(status="CLARIFY", reason="p_jj_gt_clarify")
        else:
            if s1 >= th.accept_score:
                decision = SearchDecision(status="ACCEPT", reason="score_ge_accept")
            elif s1 >= th.clarify_score:
                decision = SearchDecision(status="CLARIFY", reason="score_ge_clarify")
            else:
                decision = SearchDecision(status="REJECT", reason="score_lt_clarify")

        # 2) Tighten ACCEPT with coverage + margin gates.
        if decision.status == "ACCEPT":
            min_cov = th.min_coverage if has_nn else th.min_coverage_no_nn
            tau_margin = th.tau_margin if has_nn else th.tau_margin_no_nn
            if has_jj and coverage_top1 < float(min_cov):
                decision = SearchDecision(status="CLARIFY", reason=f"{decision.reason}+coverage_lt_min")
            elif margin_ratio < float(tau_margin):
                decision = SearchDecision(status="CLARIFY", reason=f"{decision.reason}+margin_lt_tau")

        # 3) Hard floor: don't surface very low-score matches.
        if decision.status in ("ACCEPT", "CLARIFY") and s1 < float(th.clarify_score):
            decision = SearchDecision(status="REJECT", reason="score_lt_clarify")

        # Build results (confidence as softmax probability among Top-K)
        top_rows = scored[: max(1, cfg.top_k)]
        totals = np.asarray([r[1] for r in top_rows], dtype=np.float32)
        temp = 0.2
        x = totals / max(temp, 1e-6)
        x = x - float(np.max(x))
        probs = np.exp(x)
        probs = probs / float(np.sum(probs) or 1.0)

        out: list[ScoredItem] = []
        for rank, row in enumerate(top_rows):
            p, total, s_nn, s_nn_vec, s_nn_bm25, s_jj, nn_matches, jj_matches = row
            item = idx.items[p].item
            conf = float(probs[rank]) if rank < probs.shape[0] else 0.0
            explain = None
            if req.debug:
                m_name_comparisons: int | None = None
                if valid_nns and idx.neg_name.samples:
                    m_phrases = 0
                    for p2 in cand_list:
                        it2 = idx.items[p2]
                        if it2.type_phrase_count > 0:
                            m_phrases += max(1, int(it2.type_phrase_count))
                        else:
                            m_phrases += max(1, len(it2.name_phrases))
                    m_name_comparisons = max(1, m_phrases * max(1, len(valid_nns)))

                p_nn_row = None
                if m_name_comparisons is not None and idx.neg_name.samples:
                    p_nn_row = idx.neg_name.tail_p_max(float(s_nn_vec), m_name_comparisons)

                p_jj_row = None
                if jjs and idx.neg_desc.samples:
                    m_labels = max(1, len(idx.items[p].desc_labels))
                    p_prod = 1.0
                    any_used = False
                    for m in jj_matches:
                        if (m.get("used") or 0.0) <= 0:
                            continue
                        any_used = True
                        sim = float(m.get("sim") or 0.0)
                        p_prod *= max(1e-12, idx.neg_desc.tail_p_max(sim, m_labels))
                    p_jj_row = p_prod if any_used else 1.0

                explain = MatchExplain(
                    s_total=total,
                    s_nn=s_nn,
                    s_nn_vec=s_nn_vec,
                    s_nn_bm25=s_nn_bm25,
                    s_jj=s_jj,
                    coverage=float(
                        (
                            sum(float(m.get("w") or 0.0) for m in jj_matches if float(m.get("used") or 0.0) > 0.0)
                            / max(1e-12, sum(float(m.get("w") or 0.0) for m in jj_matches))
                        )
                        if jj_matches
                        else 0.0
                    ),
                    matched_jj=tuple(jj_matches),
                    matched_nn=tuple(nn_matches),
                    margin_ratio=margin_ratio,
                    p_nn=p_nn_row,
                    p_jj=p_jj_row,
                )
            out.append(ScoredItem(id=item.id, name=item.name, score=total, confidence=conf, explain=explain))

        if decision.status == "ACCEPT":
            best = out[0]
            alternatives = tuple(out[1:])
        elif decision.status == "CLARIFY":
            best = out[0]
            alternatives = tuple(out[1:])
        else:
            best = None
            alternatives = tuple(out)

        return SearchResult(decision=decision, parsed=parsed, best=best, alternatives=alternatives, p_spurious=p_spurious_top1)
