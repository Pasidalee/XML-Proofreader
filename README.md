# XML Proofreader

A small CLI that uses a GenAI model to proofread the `<p>` elements of an XML file and wrap each mistake in an `<error>` tag.

```xml
<!-- input -->
<p>My name is john.</p>
<!-- output -->
<p>My name is <error type="capitalization" correction="John" reason="Names should be capitalized.">john</error>.</p>
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Then open `.env` and put your OpenAI key in it:

```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
python main.py samples/input.xml --lang en
```

Output is written next to the input as `input.corrected.xml`.

| Option | Description |
|---|---|
| `--lang` | BCP-47 tag, e.g. `en`, `en-GB`, `fr` (required) |
| `--no-reason` | Don't add the `reason` attribute |
| `-v` | Debug logging |

At the end it prints a summary:

```
---- Summary ----
File:            samples/input.xml
Output:          samples/input.corrected.xml
Paragraphs:      5
API calls:       5 (0 failed)
Errors tagged:   15
Time:            2.41s
Peak memory:     3.85 MB
```

## How it works

```
main.py          entry point: argument parsing, validation, exit codes, timing and memory summary
proofreader/
  processor.py   parse XML, collect <p>, call the model in parallel, write output
  llm_client.py  prompt and API call (Structured Outputs, so the response always matches a JSON schema)
  annotator.py   finds each error in the text and inserts <error> tags
```

1. **Parse** with `lxml` (keeps attributes, namespaces, comments, CDATA and whitespace).
2. **Collect** every `<p>` in document order with `root.iter()`, so nested ones are included.
3. **Proofread** each paragraph with one API call. Calls run in a thread pool, and duplicate paragraphs are only sent once.
4. **Locate** each error. The model returns the exact `original` text plus a `before` field. The `before` text is needed for errors like a missing comma, where `original` is just `" "` and appears many times in the sentence. Errors are searched left to right, so they can't overlap.
5. **Inject** `<error>` tags that wrap the original text exactly.
6. **Verify** the length invariant: the paragraph's text without tags must equal the original text. If not, the paragraph is reverted to the original.

### Design decisions

- **Prompt.** Step-by-step procedure, a checklist of easy-to-miss errors, and one example plus a no-error case.
- **Structured output.** A strict JSON schema guarantees valid replies. `temperature=0` keeps runs stable.
- **Fail safe.** A single failed call leaves only that paragraph unchanged.
- **Untouched when correct.** Paragraphs with no errors are byte-identical to the input.
- **Locale-aware.** `--lang` sets the language rules in the prompt.
- 
## Tests

```bash
python -m pytest
```

Tests use a fake model, so no API key is needed.

## Limitations / next steps

- `<p>` elements that contain child tags (e.g. `<p>Hello <b>world</b></p>`) are skipped with a warning.
- Model quality decides which errors are found; results can vary slightly between runs.
- For very large files, batching several paragraphs into one request would reduce the number of API calls.
