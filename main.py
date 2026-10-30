import argparse
import logging
import os
import re
import sys
import time
import tracemalloc
from pathlib import Path

from dotenv import load_dotenv
from lxml import etree

from proofreader.llm_client import AuthError, LLMClient
from proofreader.processor import ProofreadError, process_file

log = logging.getLogger("proofreader")

# BCP-47 check: "en", "fr", "en-US", "zh-Hant-TW" ...
LANG_PATTERN = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]{2,8})*$")

OK, FAILED, BAD_INPUT = 0, 1, 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Proofread <p> elements in an XML file using a GenAI model.")
    parser.add_argument("input", type=Path, help="Path to the input XML file")
    parser.add_argument("--lang", required=True, help="BCP-47 language tag, e.g. en, fr")
    parser.add_argument("--no-reason", action="store_true", help="Don't add the reason attribute")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def validate(args) -> bool:
    if not LANG_PATTERN.match(args.lang):
        log.error("Invalid --lang %r, expected a BCP-47 tag like 'en' or 'fr'", args.lang)
        return False
    if not args.input.is_file():
        log.error("Input file not found: %s", args.input)
        return False
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY is not set. Add it to a .env file or export it.")
        return False
    return True


def print_summary(input_path: Path, stats: dict, elapsed: float, peak_bytes: int) -> None:
    print("---- Summary ----")
    print(f"File:            {input_path}")
    print(f"Output:          {stats['output']}")
    print(f"Paragraphs:      {stats['paragraphs']}")
    print(f"API calls:       {stats['api_calls']} ({stats['failed_calls']} failed)")
    print(f"Errors tagged:   {stats['errors']}")
    print(f"Time:            {elapsed:.2f}s")
    print(f"Peak memory:     {peak_bytes / 1024 / 1024:.2f} MB")


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not validate(args):
        return BAD_INPUT

    tracemalloc.start()
    start = time.perf_counter()
    try:
        llm = LLMClient(lang=args.lang)
        stats = process_file(args.input, llm, add_reason=not args.no_reason)
    except AuthError as e:
        log.error("%s. Stopping - no output written.", e)
        return FAILED
    except ProofreadError as e:
        log.error("%s", e)
        return FAILED
    except etree.XMLSyntaxError as e:
        log.error("Invalid XML in %s: %s", args.input, e)
        return BAD_INPUT
    except Exception:
        log.exception("Unexpected error")
        return FAILED
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    print_summary(args.input, stats, elapsed, peak)
    return OK


if __name__ == "__main__":
    sys.exit(main())
