from types import SimpleNamespace

import pytest
from lxml import etree

from proofreader.annotator import annotate_paragraph, locate_errors
from proofreader.llm_client import extract_errors
from proofreader.processor import ProofreadError, process_file


INPUT_XML = """<?xml version='1.0' encoding='utf-8'?>
<doc xmlns:x="http://example.com/x">
    <title x:id="t1">Weekly notes</title>
    <p id="a">our team meets on mondays in berlin germany.</p>
    <p id="b">The report is ready.</p>
    <section>
        <p id="c">She dont like coffee tea or juice.</p>
    </section>
    <code><![CDATA[if (a < b) { return; }]]></code>
</doc>
"""

EXPECTED_XML = """<?xml version='1.0' encoding='utf-8'?>
<doc xmlns:x="http://example.com/x">
    <title x:id="t1">Weekly notes</title>
    <p id="a"><error type="capitalization" correction="Our">our</error> team meets on <error type="capitalization" correction="Mondays">mondays</error> in <error type="capitalization" correction="Berlin, Germany">berlin germany</error>.</p>
    <p id="b">The report is ready.</p>
    <section>
        <p id="c">She <error type="grammar" correction="doesn't">dont</error> like coffee<error type="punctuation" correction=", "> </error>tea<error type="punctuation" correction=", "> </error>or juice.</p>
    </section>
    <code><![CDATA[if (a < b) { return; }]]></code>
</doc>
"""


def err(type_, before, original, correction):
    return {"type": type_, "before": before, "original": original, "correction": correction}


FAKE_RESPONSES = {
    "our team meets on mondays in berlin germany.": [
        err("capitalization", "", "our", "Our"),
        err("capitalization", "meets on ", "mondays", "Mondays"),
        err("capitalization", "mondays in ", "berlin germany", "Berlin, Germany"),
    ],
    "She dont like coffee tea or juice.": [
        err("grammar", "She ", "dont", "doesn't"),
        err("punctuation", "like coffee", " ", ", "),
        err("punctuation", "coffee tea", " ", ", "),
    ],
}


class FakeLLM:
    def find_errors(self, text):
        return [dict(e) for e in FAKE_RESPONSES.get(text, [])]


class FailingLLM:
    def find_errors(self, text):
        return None  # every call failed


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "input.xml"
    path.write_text(INPUT_XML, encoding="utf-8")
    return path


# ---- whole file ----

def test_file_is_annotated_and_everything_else_preserved(input_file):
    stats = process_file(input_file, FakeLLM(), add_reason=False)

    assert stats["output"] == input_file.with_name("input.corrected.xml")
    assert stats["errors"] == 6
    assert stats["output"].read_text(encoding="utf-8").strip() == EXPECTED_XML.strip()


def test_text_is_unchanged_after_annotation(input_file):
    """without the <error> tags every <p> has its original text."""
    stats = process_file(input_file, FakeLLM())
    before = [("".join(p.itertext())) for p in etree.parse(str(input_file)).iter("p")]
    after = [("".join(p.itertext())) for p in etree.parse(str(stats["output"])).iter("p")]
    assert before == after


def test_all_calls_failed_raises_and_writes_nothing(input_file):
    with pytest.raises(ProofreadError):
        process_file(input_file, FailingLLM())
    assert not input_file.with_name("input.corrected.xml").exists()


# ---- single paragraph ----

def test_length_invariant_holds():
    p = etree.fromstring("<p>My name is john.</p>")
    assert annotate_paragraph(p, [err("capitalization", "is ", "john", "John")]) == 1
    assert "".join(p.itertext()) == "My name is john."
    assert etree.tostring(p).decode() == (
        '<p>My name is <error type="capitalization" correction="John">john</error>.</p>'
    )


def test_no_errors_leaves_paragraph_unchanged():
    p = etree.fromstring('<p id="x">All good here.</p>')
    assert annotate_paragraph(p, []) == 0
    assert etree.tostring(p).decode() == '<p id="x">All good here.</p>'


def test_reason_attribute_added():
    p = etree.fromstring("<p>My name is john.</p>")
    e = err("capitalization", "is ", "john", "John")
    e["reason"] = "Names are capitalized."
    annotate_paragraph(p, [e])
    assert p[0].get("reason") == "Names are capitalized."


# ---- locating errors ----

def test_single_space_uses_before_context():
    spans = locate_errors("We sell pens paper and ink.", [err("punctuation", "sell pens", " ", ", ")])
    assert spans[0][0] == len("We sell pens")


def test_unknown_original_is_skipped():
    assert locate_errors("Hello world.", [err("spelling", "", "wrold", "world")]) == []


def test_wrong_before_on_a_space_is_skipped():
    text = "we met in oslo norway? It was cold."
    assert locate_errors(text, [err("punctuation", "oslo norway ", " ", ", ")]) == []


# ---- model response ----

def _fake_response(content, refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_extract_errors_from_json_reply():
    reply = '{"corrected_text": "b", "errors": [{"type": "grammar", "original": "a", "correction": "b"}]}'
    assert extract_errors(_fake_response(reply))[0]["correction"] == "b"
