"""Pure-logic unit tests (no servers, no audio)."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from pipeline import textutils as tu


@pytest.mark.parametrize("text,expected", [
    ("Hey Jarvis.", ""),
    ("Hey, Jarvis, what is the capital of France?", "what is the capital of France?"),
    ("hey travis what time is it", "what time is it"),
    ("Jarvis", ""),
    ("what is jarvis in iron man", "what is jarvis in iron man"),   # not a wake phrase mid-sentence
    ("Hey Jarvis, what is 2+3?", "what is 2+3?"),
    ("Darvis, what is 2 plus 3?", "what is 2 plus 3?"),
    ("Harvest what is 2 plus 3", "what is 2 plus 3"),
    ("Hey, Jar this, stop.", "stop."),
    ("Hey Service what time is it", "what time is it"),
    ("Is Jarvis what is the weather", "what is the weather"),
    ("Harvey Weinstein was a producer", "Harvey Weinstein was a producer"),
])
def test_strip_wake_phrase(text, expected):
    assert tu.strip_wake_phrase(text, "hey_jarvis") == expected


@pytest.mark.parametrize("text,junk", [
    ("[BLANK_AUDIO]", True), ("*sigh*", True), ("(wind blowing)", True), ("", True), (None, True), (".", True),
    ("you", True), ("Thanks for watching!", True),
    ("What is the capital of France?", False), ("Thank you.", False), ("stop", False), ("(wind) hello there", False),
])
def test_is_hallucination(text, junk):
    assert tu.is_hallucination(text) is junk


@pytest.mark.parametrize("text,intent", [
    ("stop", "stop"), ("Okay, stop it.", "stop"), ("never mind", "stop"), ("Cancel.", "stop"),
    ("goodbye", "sleep"), ("That's all for now", "sleep"), ("go to sleep", "sleep"),
    ("repeat that", "repeat"), ("Can you say that again?", "repeat"), ("what did you say", "repeat"),
    ("what time is it?", "time"), ("What's the date", "date"), ("hello", "greeting"), ("thanks", "thanks"),
    # must NOT trigger instant intents:
    ("I want to stop at the store, what's open?", None),
    ("hello can you explain relativity to me please in detail", None),
    ("how do I stop a leaking tap", None),
    ("what time does the train to London leave tomorrow morning", None),
])
def test_classify_intent(text, intent):
    assert tu.classify_intent(text) == intent


def test_answer_intent_time_and_date():
    t = datetime(2026, 10, 8, 14, 5)
    assert tu.answer_intent("time", t) == "It's 2:05 PM."
    assert tu.answer_intent("date", t) == "Today is Thursday, October 8."
    assert tu.answer_intent("stop", t) is None


def _chunks(text, step=3):
    c = tu.SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += c.feed(text[i:i + step])
    return out + c.flush()


def test_chunker_sentences_abbreviations_decimals():
    out = _chunks("Sure! Dr. Smith said pi is 3.14, which is roughly right. Here is another long sentence, ok? Done")
    assert out[0].startswith("Sure! Dr. Smith")        # "Sure!" too short alone, "Dr." not a boundary
    assert "3.14" in out[0]
    assert out[-1] == "Done"


def test_chunker_is_independent_of_token_boundaries():
    text = "The capital of France is Paris. It is known for the Eiffel Tower. Many people visit every year."
    assert _chunks(text, 1) == _chunks(text, 7) == _chunks(text, 100)


def test_chunker_first_clause_release_for_low_latency():
    c = tu.SentenceChunker(first_clause_words=10)
    out = []
    for w in "Well, to answer that question properly I would say that the matter is complicated, because many things apply".split():
        out += c.feed(w + " ")
    assert out and out[0].endswith("complicated,")       # released before any full stop arrived


def test_clean_for_speech_strips_markdown_and_emoji():
    s = tu.clean_for_speech("**Hello** there 😀! Visit https://x.com `now`:\n- item one\n- item two")
    assert "*" not in s and "😀" not in s and "http" not in s and "`" not in s


def test_trim_to_sentence_and_memory():
    assert tu.trim_to_sentence("It is sunny. It will be wa") == "It is sunny."
    assert tu.trim_to_sentence("All good.") == "All good."
    assert tu.trim_for_memory("One. Two is longer than the limit.", 12) == "One."


@pytest.mark.parametrize("text,expected", [
    ("okay", "stop"), ("got it", "stop"), ("no no", "stop"), ("wait", "stop"), ("yeah", "stop"),
    ("what is the capital of France", None), ("why", None),
])
def test_cut_in_acknowledgements_only_count_when_interrupting(text, expected):
    assert tu.classify_intent(text, barge=True) == expected
    assert tu.classify_intent(text, barge=False) in (None, "stop")      # "okay" alone is not a command when idle
    if expected == "stop" and text not in ("wait",):
        assert tu.classify_intent(text, barge=False) is None


def test_repetition_loops_are_hallucinations():
    assert tu.is_hallucination("I'll tell you, I'll tell you, I'll tell you, I'll tell you, I'll tell you, I'll tell you")
    assert tu.is_hallucination("thank you thank you thank you thank you thank you thank you thank you thank you")
    assert not tu.is_hallucination("no no no what I meant was the other one")
    assert not tu.is_hallucination("what is the capital of France")
