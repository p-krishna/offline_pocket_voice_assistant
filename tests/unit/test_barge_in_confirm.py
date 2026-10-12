"""
The decision taken after speech over our answer (Pipeline._confirm_interject): resume on noise, stop on
a muffled short word, stop-and-answer on real speech. Pipeline.__init__ loads models, so the object is built
without it and only the attributes the method touches are provided.
"""
from types import SimpleNamespace

import pytest

from pipeline.assistant_pipeline import Pipeline, Utterance
from stt.whisper_cpp import STTResult


class FakeSpeaker:
    def __init__(self):
        self.calls = []

    def resume(self, why=""):
        self.calls.append(("resume", why))

    def stop(self, why=""):
        self.calls.append(("stop", why))


def make(text, logprob=-0.1, last_response="Mars is a planet.", paused=True, speech_ms=600, wakeish=False):
    p = object.__new__(Pipeline)
    p.cfg = SimpleNamespace(wakeword="hey_jarvis", stt_min_logprob=-0.7, log_transcripts=True)
    p.speaker = FakeSpeaker()
    p._last_response = last_response
    p._m_interrupt_detected = p._m_interrupt_success = 0
    queued = []
    p._enqueue_utterance = queued.append
    res = STTResult(text, avg_logprob=logprob)
    utt = Utterance([], "interject", speech_ms, 0.05, 0.0, 0.0, text=text, stt=res, wakeish=wakeish, paused=paused)
    return p, utt, queued


def test_noise_resumes_the_answer():
    p, utt, queued = make("[BLANK_AUDIO]")
    p._confirm_interject(utt)
    assert p.speaker.calls == [("resume", "interject_noise")]
    assert not queued


def test_whisper_junk_resumes_the_answer():
    p, utt, queued = make("Thanks for watching.")
    p._confirm_interject(utt)
    assert p.speaker.calls == [("resume", "interject_noise")]


def test_clear_request_stops_and_is_queued():
    p, utt, queued = make("What is the capital of Spain?")
    p._confirm_interject(utt)
    assert p.speaker.calls == [("stop", "interject")]
    assert queued == [utt]


def test_muffled_short_word_is_a_silent_stop():
    p, utt, queued = make("Snobber.", logprob=-1.4, speech_ms=400)
    p._confirm_interject(utt)
    assert p.speaker.calls == [("stop", "interject_unclear")]
    assert not queued


def test_long_unclear_speech_stops_and_is_handled_as_a_turn():
    p, utt, queued = make("mumble mumble mumble", logprob=-1.4, speech_ms=3000)
    p._confirm_interject(utt)
    assert p.speaker.calls == [("stop", "interject")]
    assert queued == [utt]


def test_bye_is_speech_even_though_whisper_hallucinates_it():
    p, utt, queued = make("Bye.")
    p._confirm_interject(utt)
    assert p.speaker.calls == [("stop", "interject")]
    assert queued == [utt]


def test_unclear_while_thinking_keeps_the_pending_answer():
    p, utt, queued = make("Snobber.", logprob=-1.4, paused=False, speech_ms=400)
    p._confirm_interject(utt)
    assert p.speaker.calls == []
    assert not queued


def test_stt_failure_resumes_instead_of_staying_silent():
    p, utt, queued = make("x")
    utt.text = None
    utt.stt_future = None

    def boom(_):
        raise RuntimeError("stt down")
    p._get_transcript = boom
    p._confirm_interject(utt)
    assert p.speaker.calls == [("resume", "interject_stt_failed")]
