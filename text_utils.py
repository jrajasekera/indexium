from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

MIN_FRAGMENT_LENGTH = 4


def calculate_top_text_fragments(
    entries: Iterable[Tuple[str, int]],
    top_n: int,
    min_length: int = MIN_FRAGMENT_LENGTH,
) -> List[Dict[str, object]]:
    """Return the top substrings by frequency and length from weighted text entries."""
    if not top_n or top_n <= 0:
        return []

    substring_data: Dict[str, Dict[str, object]] = {}
    minimum = max(MIN_FRAGMENT_LENGTH, min_length)

    for raw_text, weight in entries:
        if not raw_text:
            continue
        weight = max(int(weight or 0), 0)
        if weight == 0:
            weight = 1

        normalized = raw_text.lower()
        if len(normalized) < minimum:
            continue

        per_string_seen: set[str] = set()
        length = len(normalized)
        for start in range(length - minimum + 1):
            for end in range(start + minimum, length + 1):
                substring_lower = normalized[start:end]
                if substring_lower in per_string_seen:
                    continue
                per_string_seen.add(substring_lower)

                display_slice = raw_text[start:end]
                info = substring_data.get(substring_lower)
                if info is None:
                    info = {
                        "count": 0,
                        "length": end - start,
                        "display": display_slice,
                    }
                    substring_data[substring_lower] = info

                info["count"] = int(info["count"]) + weight
                if (end - start) > info.get("length", 0):
                    info["length"] = end - start
                    info["display"] = display_slice

    if not substring_data:
        return []

    ranked = sorted(
        (
            {
                "substring": data["display"],
                "lower": key,
                "count": int(data["count"]),
                "length": int(data["length"]),
            }
            for key, data in substring_data.items()
        ),
        key=lambda item: (-item["count"], -item["length"], item["lower"]),
    )

    return ranked[:top_n]
