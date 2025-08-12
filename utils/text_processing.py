import re
from typing import List, Dict, Callable, Set

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def sentence_split(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]

def estimate_tokens(text: str) -> int:
    # super rough ~4 chars/token
    return max(1, len(text) // 4)

def _normalize_label(label: str) -> str:
    # "B-per" -> "per", "I-geo" -> "geo", "O" stays "O"
    return label.split("-", 1)[-1] if "-" in label else label

def chunk_by_entities(
    text: str,
    ner_sentence_entities: Callable[[str], List[Dict]],
    max_tokens_per_chunk: int = 2200,
    hard_cap: int = 2600,
) -> List[Dict]:
    """
    Returns list of chunks:
      { "text": str, "sentences": [str, ...], "entities": [str, ...] }
    `ner_sentence_entities(sent)` must return list[{"entity_group","word","score"}].
    """
    sents = sentence_split(text)
    chunks = []
    cur_sents: List[str] = []
    cur_ents: Set[str] = set()
    cur_tokens = 0

    def ents_for_sentence(s: str) -> Set[str]:
        ents = set()
        for item in ner_sentence_entities(s):
            lab = item.get("entity_group", "O")
            if lab != "O":
                ents.add(_normalize_label(lab))
        return ents

    def push_chunk():
        if not cur_sents:
            return
        chunk_text = " ".join(cur_sents).strip()
        chunks.append({
            "text": chunk_text,
            "sentences": cur_sents.copy(),
            "entities": sorted(cur_ents),
        })

    def should_split(next_sent: str, next_ents: Set[str]) -> bool:
        nonlocal cur_tokens
        # token budget
        if cur_tokens + estimate_tokens(next_sent) > max_tokens_per_chunk:
            return True
        # topic shift (low entity overlap) once chunk is reasonably sized
        if cur_tokens > int(max_tokens_per_chunk * 0.6):
            inter = len(cur_ents & next_ents)
            uni = len(cur_ents | next_ents) or 1
            jacc = inter / uni
            if jacc < 0.2:
                return True
        return False

    for s in sents:
        e = ents_for_sentence(s)
        if cur_sents and should_split(s, e):
            push_chunk()
            cur_sents.clear()
            cur_ents.clear()
            cur_tokens = 0

        cur_sents.append(s)
        cur_ents |= e
        cur_tokens += estimate_tokens(s)

        # super hard cap splitter
        while cur_tokens > hard_cap and len(cur_sents) > 1:
            mid = len(cur_sents) // 2
            left = " ".join(cur_sents[:mid]).strip()
            chunks.append({
                "text": left,
                "sentences": cur_sents[:mid],
                "entities": sorted(cur_ents),
            })
            cur_sents = cur_sents[mid:]
            cur_tokens = sum(estimate_tokens(x) for x in cur_sents)

    push_chunk()
    return chunks
