import json
import logging
import os

import openai

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"

LANGUAGE_NAMES = {
    "en": "English", "en-us": "American English", "en-gb": "British English",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch",
}

ERROR_TYPES = ["grammar", "spelling", "punctuation", "capitalization", "clarity"]

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "proofread_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "corrected_text": {"type": "string"},
                "errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "correction": {"type": "string"},
                            "before": {"type": "string"},
                            "type": {"type": "string", "enum": ERROR_TYPES},
                            "reason": {"type": "string"},
                        },
                        "required": ["original", "correction", "before", "type", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["corrected_text", "errors"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are an expert proofreader for {language} (language tag: {lang}).
Find EVERY real error in the paragraph, using the rules of {language} - not the rules of any other language.

## Procedure
1. Write "corrected_text": the paragraph with every error fixed, nothing else changed.
2. Compare it with the paragraph word by word. Each difference is one item in "errors",
   in order of position.

## Check carefully
- Capitalization, spelling (including accents, diacritics, apostrophes) and agreement
  (verb, gender, number, case) as {language} requires.
- Confused words, decided by meaning.
- Punctuation: commas, end marks, and the spacing and quote conventions of {language}.
- Every occurrence of a repeated mistake.

## Do not report
- Correct text, style or word choice, or valid regional spellings.
- A problem another of your corrections already fixes.
- Clarity, unless the wording is genuinely confusing.

## Fields
- "original": the exact wrong text from the paragraph - the text being replaced, never
  a neighbouring word. Smallest span, never empty.
- "correction": the text that replaces "original".
- "before": the 1-3 words immediately before "original", copied exactly with spaces
  ("" at the start).
- "type": spelling, grammar, punctuation, capitalization or clarity.
- "reason": one short sentence in {language}.
- Missing punctuation between words: "original" is the space (" "), "correction" is
  the punctuation plus the space (", "). Never attach it to a word.
- Items must not overlap. Applying all corrections must give corrected_text exactly.

## Example (English, to show the format only)

Text: our shop in rome italy sells tea coffee and cake.
{"corrected_text": "Our shop in Rome, Italy sells tea, coffee, and cake.", "errors": [
 {"original": "our", "correction": "Our", "before": "", "type": "capitalization", "reason": "A sentence starts with a capital letter."},
 {"original": "rome italy", "correction": "Rome, Italy", "before": "shop in ", "type": "capitalization", "reason": "Place names are capitalized, and a city is separated from its country by a comma."},
 {"original": " ", "correction": ", ", "before": "sells tea", "type": "punctuation", "reason": "Items in a list are separated by commas."},
 {"original": " ", "correction": ", ", "before": "tea coffee", "type": "punctuation", "reason": "A comma goes before 'and' in a list of three or more items."}
]}

Text: The quarterly results were published on Friday, and the board approved the new plan.
{"corrected_text": "The quarterly results were published on Friday, and the board approved the new plan.", "errors": []}"""


class AuthError(Exception):
    """API key is missing, wrong or has no access."""


class LLMClient:
    def __init__(self, lang: str):
        self.lang = lang
        self.language = LANGUAGE_NAMES.get(lang.lower(), lang)
        self.model = os.getenv("PROOFREAD_MODEL", DEFAULT_MODEL)
        self.client = openai.OpenAI(max_retries=3, timeout=60)
        self.system_prompt = SYSTEM_PROMPT.replace("{language}", self.language).replace("{lang}", lang)

    def find_errors(self, text: str) -> list[dict] | None:
        """Send text to the model and return the errors it finds.

        Returns a list of errors ([] = text has no errors), or None if the call failed.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=2000,
                response_format=RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Text: {text}"},
                ],
            )
            return extract_errors(response)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise AuthError(f"Authentication failed - check OPENAI_API_KEY ({e.status_code})") from e
        except openai.APIError as e:
            logger.error("API call failed for text %r: %s", text[:40], e)
        except ValueError as e:
            logger.error("Unexpected model response for text %r: %s", text[:40], e)
        return None


def extract_errors(response) -> list[dict]:
    """Get the errors list from the model's JSON reply."""
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise ValueError(f"model refused: {message.refusal}")
    if not message.content:
        raise ValueError("empty response")
    errors = json.loads(message.content)["errors"]
    if not isinstance(errors, list):
        raise ValueError("'errors' is not a list")
    return errors
