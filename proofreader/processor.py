import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

from .annotator import annotate_paragraph

logger = logging.getLogger(__name__)

MAX_WORKERS = 8

class ProofreadError(Exception):
    """The run could not produce a useful result."""


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.corrected{input_path.suffix}")


def write_tree(tree: etree._ElementTree, out_path: Path) -> None:
    """Write the document keeping the same style of XML declaration as the input"""
    encoding = (tree.docinfo.encoding or "utf-8").lower()
    body = etree.tostring(tree, encoding=encoding, xml_declaration=False)
    header = f"<?xml version='{tree.docinfo.xml_version or '1.0'}' encoding='{encoding}'?>\n"
    with open(out_path, "wb") as f:
        f.write(header.encode(encoding))
        f.write(body)


def find_paragraphs(root: etree._Element) -> list[etree._Element]:
    """Extract all <p> elements in document order"""
    paragraphs = []
    for el in root.iter():
        if not isinstance(el.tag, str) or etree.QName(el).localname != "p":
            continue
        if len(el):
            logger.warning("Skipping <p> with child elements (line %s)", el.sourceline)
            continue
        if el.text and el.text.strip():
            paragraphs.append(el)
    return paragraphs


def process_file(input_path: Path, llm, add_reason: bool = True) -> dict:
    """Proofread all <p> elements and write to output file`.

    Raises ProofreadError if every API call fails. Returns run stats.
    """
    parser = etree.XMLParser(strip_cdata=False, remove_blank_text=False)
    tree = etree.parse(str(input_path), parser)
    paragraphs = find_paragraphs(tree.getroot())
    logger.info("Found %d <p> elements to proofread", len(paragraphs))

    # If same sentence appearing twice, only one API call.
    unique_texts = list(dict.fromkeys(p.text for p in paragraphs))

    pool = ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(unique_texts))))
    try:
        results = dict(zip(unique_texts, pool.map(llm.find_errors, unique_texts)))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    failed = sum(1 for r in results.values() if r is None)
    if unique_texts and failed == len(unique_texts):
        raise ProofreadError(f"All {failed} API calls failed - no output written. See errors above.")
    if failed:
        logger.warning("%d of %d API calls failed; those paragraphs were left unchanged",
                       failed, len(unique_texts))

    total_errors = 0
    for p in paragraphs:
        # copy the dicts so a shared result isn't modified twice
        errors = [dict(e) for e in results[p.text] or []]
        total_errors += annotate_paragraph(p, errors, add_reason=add_reason)

    out_path = output_path_for(input_path)
    write_tree(tree, out_path)
    return {
        "output": out_path,
        "paragraphs": len(paragraphs),
        "api_calls": len(unique_texts),
        "failed_calls": failed,
        "errors": total_errors,
    }
