import logging

from lxml import etree

logger = logging.getLogger(__name__)

VALID_TYPES = {"grammar", "spelling", "punctuation", "capitalization", "clarity"}


def locate_errors(text: str, errors: list[dict]) -> list[tuple[int, int, dict]]:
    """Find where each error is in text.

    Returns (start, end, error) tuples.
    Errors that cannot be found exactly are dropped (logged).
    """
    spans = []
    cursor = 0
    for err in errors:
        original = err.get("original") or ""
        correction = err.get("correction")
        if not original or correction is None or original == correction:
            continue
        if err.get("type") not in VALID_TYPES:
            err["type"] = "clarity"

        start = _find(text, original, err.get("before") or "", cursor)
        if start == -1:
            logger.warning("Skipping error, %r not found in text %r", original, text[:40])
            continue
        end = start + len(original)
        spans.append((start, end, err))
        cursor = end
    return spans


def _find(text: str, original: str, before: str, cursor: int) -> int:
    """Search with the 'before' first (needed for errors like a single
    space), then fall back to searching for the original text.
    """
    if before:
        idx = text.find(before + original, max(cursor - len(before), 0))
        if idx != -1 and idx + len(before) >= cursor:
            return idx + len(before)
    elif cursor == 0 and text.startswith(original):
        # Empty 'before' means the error is at the very start of the text.
        return 0
    rest = text[cursor:]
    if rest.count(original) == 1:
        return cursor + rest.index(original)
    return -1


def annotate_paragraph(p: etree._Element, errors: list[dict], add_reason: bool = True) -> int:
    """Rewrite a text-only <p> with <error> tags. Returns number of tags added.

    Leaves the element unchanged if there's nothing to change.
    """
    text = p.text or ""
    spans = locate_errors(text, errors)
    if not spans:
        return 0

    # Build the new content.
    p.text = text[:spans[0][0]]
    for i, (start, end, err) in enumerate(spans):
        el = etree.SubElement(p, "error")
        el.set("type", err["type"])
        el.set("correction", err["correction"])
        if add_reason and err.get("reason"):
            el.set("reason", err["reason"])
        el.text = text[start:end]
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        el.tail = text[end:next_start] or None
    p.text = p.text or None

    # Text without tags must be identical to the original.
    new_text = "".join(p.itertext())
    if new_text != text:
        logger.error("Length invariant broken, reverting paragraph %r", text[:40])
        for child in list(p):
            p.remove(child)
        p.text = text
        return 0
    return len(spans)
