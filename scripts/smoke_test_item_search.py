import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from item_search import Item, ItemSearchEngine, SearchRequest


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    device = os.getenv("DEVICE", "cpu")

    items_path = Path(__file__).resolve().parents[1] / "data" / "items_sample.json"
    payload = json.loads(items_path.read_text(encoding="utf-8"))
    items_raw = payload["items"] if isinstance(payload, dict) else payload
    items = []
    for it in items_raw:
        raw_type = it.get("type")
        item_type = raw_type if raw_type else None

        # demo: multi-type support (string[]); engine should pick best-matching type phrase
        if str(it.get("id")) == "veh-01":
            item_type = ["\u5361\u8f66", "\u8f66", "\u6c7d\u8f66"]

        items.append(
            Item(
                id=str(it["id"]),
                name=str(it["name"]),
                type=item_type,
                aliases=tuple(it.get("aliases") or ()),
                desc_labels=tuple(it.get("desc_labels") or it.get("labels") or ()),
            )
        )

    model = SentenceTransformer(model_name, device=device)
    engine = ItemSearchEngine(model)
    stats = engine.load_items(items)
    print("Loaded:", stats)

    queries = [
        "破损的蓝色卡车",
        "红色的破烂东西",
        "帮我找一下铁的钥匙",
        "一个木制的箱子",
        "绿色药水",
        "\u6c7d\u8f66",
        "\u5927\u95e8",
    ]

    for q in queries:
        res = engine.search(SearchRequest(query=q, debug=True))
        print("\n===", q)
        print("decision:", res.decision.status, res.decision.reason)
        print("parsed:", {"nn": res.parsed.nn, "jj": res.parsed.jj, "head": res.parsed.head_noun})
        if res.best:
            print("best:", res.best.id, res.best.name, "score=", res.best.score, "conf=", res.best.confidence)
            if res.best.explain:
                print("p_nn:", res.best.explain.p_nn, "p_jj:", res.best.explain.p_jj, "margin:", res.best.explain.margin_ratio)
                print("nn:", res.best.explain.matched_nn)
                print("jj:", res.best.explain.matched_jj)
        else:
            print("best: None")
        if res.alternatives:
            print("alts:", [(a.id, a.name, round(a.score, 3), round(a.confidence, 3)) for a in res.alternatives])


if __name__ == "__main__":
    main()
