"""Regression tests for the 2026-10-13 review findings F1, F7–F10 (text logic)."""
import pytest

from pipeline import textutils as tu


# F7 — acknowledgements and goodbyes are not "hallucinations"
@pytest.mark.parametrize("text", ["Okay.", "okay", "OK", "Ok."])
def test_okay_is_not_junk(text):
    assert not tu.is_hallucination(text)


def test_okay_is_a_cut_in_stop_only_over_the_assistant():
    assert tu.classify_intent("Okay.", barge=True) == "stop"
    assert tu.classify_intent("Okay.", barge=False) is None


def test_whisper_silence_hallucinations_are_still_junk():
    for t in ["Thanks for watching.", "you", "Bye.", "[BLANK_AUDIO]", "the end"]:
        assert tu.is_hallucination(t), t


def test_bye_is_still_a_sleep_intent():
    assert tu.classify_intent("Bye.") == "sleep"


# F8 — "yes" over a question is an answer
@pytest.mark.parametrize("text", ["Yes", "yeah", "sure", "okay", "no"])
def test_affirmatives_stop_normally_but_answer_a_question(text):
    assert tu.classify_intent(text, barge=True) == "stop"
    assert tu.classify_intent(text, barge=True, awaiting_answer=True) is None


def test_explicit_stop_wins_even_when_a_question_was_asked():
    assert tu.classify_intent("stop", barge=True, awaiting_answer=True) == "stop"


def test_ends_with_question():
    assert tu.ends_with_question("Shall we try again?")
    assert tu.ends_with_question('He asked "why?" ')
    assert not tu.ends_with_question("That is correct.")
    assert not tu.ends_with_question("")


# F9 — time/date only when that is the whole question
@pytest.mark.parametrize("text,intent", [
    ("What time is it?", "time"), ("what's the time", "time"), ("Hey, what time is it right now", "time"),
    ("what's the date", "date"), ("What day is it today?", "date"),
])
def test_time_date_intents(text, intent):
    assert tu.classify_intent(text) == intent


@pytest.mark.parametrize("text", [
    "What time is it in Tokyo?", "what time is it when the sun sets",
    "What's the date of the French revolution?", "what is today's date in Japan",
])
def test_time_date_inside_a_bigger_question_goes_to_the_llm(text):
    assert tu.classify_intent(text) is None


# F10 — endpoint classification only looks for the wake phrase when it was heard
def test_endpoint_bare_wake_phrase_waits_either_way():
    assert tu.endpoint_class("Hey Jarvis.", "hey_jarvis", wakeish=True) == "wake_wait"
    assert tu.endpoint_class("Hey Jarvis.", "hey_jarvis", wakeish=False) == "wake_wait"


def test_strict_strip_needs_an_explicit_greeting():
    assert tu.strip_wake_phrase("Hey Jarvis, stop.", "hey_jarvis", strict=True) == "stop."
    assert tu.strip_wake_phrase("Hey Harvest what is 2 plus 3", "hey_jarvis", strict=True) == "what is 2 plus 3"
    assert tu.strip_wake_phrase("Jarvis what time is it", "hey_jarvis", strict=True) == "Jarvis what time is it"
    assert tu.strip_wake_phrase("Service animals in Germany", "hey_jarvis", strict=True) == "Service animals in Germany"
    # loose mode (capture known to contain the wake phrase) still handles clipped / bare mis-hearings
    assert tu.strip_wake_phrase("Harvest what is 2 plus 3", "hey_jarvis") == "what is 2 plus 3"


def test_service_is_not_eaten_without_the_wake_phrase():
    # the real-word fuzzy matches ("service", "harvest") are only safe inside a wake capture
    assert tu.endpoint_class("Service animals in Germany.", "hey_jarvis", wakeish=False) == "question"


@pytest.mark.parametrize("raw,expected", [
    (None, "default"), ("What is the capital of France?", "question"),
    ("I would like to know about the", "hesitation"), ("and then,", "hesitation"),
    ("Tell me a story", "default"), ("Explain the rules.", "question"),
    ("Ich möchte wissen und", "hesitation"),
])
def test_endpoint_class(raw, expected):
    assert tu.endpoint_class(raw, "hey_jarvis", wakeish=False) == expected


def test_endpoint_noise_after_wake_waits():
    assert tu.endpoint_class("[BLANK_AUDIO]", "hey_jarvis", wakeish=True) == "wake_wait"
    assert tu.endpoint_class("[BLANK_AUDIO]", "hey_jarvis", wakeish=False) == "default"


# F1 — German intents
@pytest.mark.parametrize("text,intent", [
    ("Danke", "thanks"), ("Vielen Dank", "thanks"), ("Tschüss", "sleep"), ("Auf Wiedersehen", "sleep"),
    ("Stopp", "stop"), ("Hör auf", "stop"), ("Noch einmal", "repeat"), ("Hallo", "greeting"),
    ("Wie spät ist es?", "time"),
])
def test_german_intents(text, intent):
    assert tu.classify_intent(text) == intent
