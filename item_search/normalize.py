from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def to_half_width(text: str) -> str:
    return (
        str(text)
        .replace("\u3000", " ")
        .translate({code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)})
    )


def normalize_text(text: str) -> str:
    import re

    s = to_half_width(text).lower()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DEFAULT_SYNONYMS: dict[str, str] = {
    "lorry": "truck",
    "truck": "卡车",
    "vehicle": "车辆",
    "car": "汽车",
    "door": "门",
    "gate": "门",
    "tree": "树",
    "pine": "松树",
    "tower": "塔",
    "watchtower": "瞭望塔",
    "metal": "金属",
    "steel": "钢制",
    "wood": "木制",
    "broken": "破损",
    "damage": "破损",
    "破烂": "破损",
    "破舊": "破旧",
    "破旧": "旧",
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    # Common shorthand colors
    "红": "红色",
    "蓝": "蓝色",
    "绿": "绿色",
    "黄": "黄色",
    "黑": "黑色",
    "白": "白色",
    "灰": "灰色",
    "紫": "紫色",
    "橙": "橙色",
    "棕": "棕色",
    "粉": "粉色",
    "house": "房子",
    "home": "房子",
    "building": "建筑",
}


DEFAULT_GENERIC_NOUNS: set[str] = {
    "东西",
    "物品",
    "道具",
    "玩意",
    "玩意儿",
    "东西儿",
    "家伙",
    "物件",
    "物",
}


DEFAULT_COLOR_LABELS: set[str] = {
    "红色",
    "蓝色",
    "绿色",
    "黄色",
    "黑色",
    "白色",
    "灰色",
    "紫色",
    "橙色",
    "棕色",
    "粉色",
}


@dataclass(frozen=True)
class Normalizer:
    synonyms: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYNONYMS))
    generic_nouns: set[str] = field(default_factory=lambda: set(DEFAULT_GENERIC_NOUNS))
    color_labels: set[str] = field(default_factory=lambda: set(DEFAULT_COLOR_LABELS))

    @staticmethod
    def from_files(
        synonyms_path: str | None = None,
        generic_nouns_path: str | None = None,
        color_labels_path: str | None = None,
    ) -> "Normalizer":
        synonyms = dict(DEFAULT_SYNONYMS)
        generic_nouns = set(DEFAULT_GENERIC_NOUNS)
        color_labels = set(DEFAULT_COLOR_LABELS)

        if synonyms_path:
            data = json.loads(Path(synonyms_path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                synonyms.update({normalize_text(k): normalize_text(v) for k, v in data.items()})

        def _load_lines(p: str) -> list[str]:
            return [
                normalize_text(line)
                for line in Path(p).read_text(encoding="utf-8").splitlines()
                if normalize_text(line)
            ]

        if generic_nouns_path:
            generic_nouns.update(_load_lines(generic_nouns_path))
        if color_labels_path:
            color_labels.update(_load_lines(color_labels_path))

        return Normalizer(synonyms=synonyms, generic_nouns=generic_nouns, color_labels=color_labels)

    def norm(self, text: str) -> str:
        n = normalize_text(text)
        if not n:
            return ""
        return self.synonyms.get(n, n)

    def is_generic_noun(self, text: str) -> bool:
        return self.norm(text) in self.generic_nouns

    def type_coef(self, label: str) -> float:
        # Changed to 1.0 to fully weight color matches (was 0.2)
        return 1.0 if self.norm(label) in self.color_labels else 1.0
