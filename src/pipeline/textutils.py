"""
Pure text helpers for the voice pipeline (no I/O → easy to unit test).

 - wake-word stripping, STT hallucination filtering
 - strict intent matching for instant answers ("what time is it", "stop", ...)
 - streaming sentence chunker that feeds TTS while the LLM is still decoding
 - speech cleanup (markdown/emoji) and memory trimming
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

# ── STT clean-up ─────────────────────────────────────────────────────────────

# Whisper tokens that mean "nothing was said" — never send these to the LLM.
BLANK_TOKENS = {"[BLANK_AUDIO]", "[SILENCE]", "[MUSIC]", "(silence)", "(ambient noise)",
                "(birds chirping)", "(chuckles)", "(speaking in foreign language)",
                "(upbeat music)", "(crowd chattering)", "(humming)", "(coughs)"}

# Phrases whisper-tiny hallucinates on near-silence / noise.
_JUNK = {
    "you", "bye", "bye.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.", "subscribe",
    "please subscribe", "the end", "the end.", "so", "uh", "um", "hmm", "mm", "mhm",
    "oh", "ah", "huh", "okay", "okay.", "ok",
}

# Compared on the normalised form ("okay." and "okay" are the same word). Plain acknowledgements are
# legitimate cut-ins ("okay" = enough), so they are never junk.
_JUNK_NORMALIZED = None   # filled below, after normalize() exists

_WRAPPED = re.compile(r"^\W*(\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪[^♪]*♪?)\W*$")


def normalize(text: str) -> str:
    """lowercase, drop punctuation, collapse whitespace (for matching only)."""
    t = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


_JUNK_NORMALIZED = {normalize(j) for j in _JUNK} - {"okay", "ok"}


def has_repetition_loop(text: str) -> bool:
    """Whisper's classic failure: "I'll tell you, I'll tell you, I'll tell you, …"."""
    w = normalize(text).split()
    if len(w) < 8:
        return False
    for n in (1, 2, 3, 4, 5):
        run, best = 1, 1
        for i in range(n, len(w) - n + 1, n):
            run = run + 1 if w[i:i + n] == w[i - n:i] else 1
            best = max(best, run)
        if best >= 4 and best * n >= 8:
            return True
    return False


def is_hallucination(text: Optional[str]) -> bool:
    """True when the transcript carries no real user speech."""
    if text is None:
        return True
    if has_repetition_loop(text):
        return True
    t = text.strip()
    if len(re.sub(r"[\W_]+", "", t)) < 2:
        return True
    if t in BLANK_TOKENS or _WRAPPED.match(t):
        return True
    # Strip any bracketed noise annotations, see if something real remains.
    rest = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", t).strip()
    if len(re.sub(r"[\W_]+", "", rest)) < 2:
        return True
    return normalize(rest) in _JUNK_NORMALIZED


def clean_transcript(text: str) -> str:
    """Remove bracketed non-speech annotations ("(wind blowing)", "*sigh*") from a transcript."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪", " ", text or "")
    return re.sub(r"\s+", " ", t).strip()


def _wake_name(wakeword: str) -> str:
    return wakeword.replace("hey_", "").replace("_", " ").strip()


def strip_wake_phrase(text: str, wakeword: str = "hey_jarvis", strict: bool = False) -> str:
    """
    Remove a leading 'Hey Jarvis' (and common whisper mis-hearings) from a transcript.

    `strict=False` is for captures known to contain the wake phrase: the greeting is optional and the fuzzy
    list may match bare ("Harvest what is 2 plus 3"). `strict=True` is for everything else: an explicit
    greeting is required, so a request that merely starts with a real word that sounds like the name
    ("Service animals in Germany…") is left alone.
    """
    name = re.escape(_wake_name(wakeword))
    if _wake_name(wakeword) == "jarvis":
        # whisper-tiny hears "Jarvis" as Darvis / Travis / Jarvus / Harvest / Jar this / service …
        fuzzy = (r"(?:jarvis|jervis|jarves|jarvus|jarvice|jarvas|darvis|darvus|travis|service|harvest|"
                 r"jar\s*this|jar\s*vis|jar\s*bus|jar\s*vice|j\w?rv\w{1,3})")
    else:
        fuzzy = name
    if strict:
        pat = re.compile(rf"^\W*(?:hey|hi|hay|okay|ok|hello)[\s,.!-]+{fuzzy}\b[\s,.!?:;-]*", re.I)
        return pat.sub("", text.strip(), count=1).strip()
    greet = r"(?:hey|hi|hay|okay|ok|hello|a|eh|it|is|that's|hate|he)"
    pat = re.compile(rf"^\W*(?:{greet}[\s,.!-]+)?{fuzzy}\b[\s,.!?:;-]*", re.I)
    out = pat.sub("", text.strip(), count=1)
    # Whisper sometimes splits it: "Hey. Jarvis, ..."
    out = re.sub(rf"^\W*hey[\s,.!-]+{fuzzy}\b[\s,.!?:;-]*", "", out, flags=re.I)
    return out.strip()


# ── Intents (strict: only near-whole-utterance matches) ──────────────────────

_PRE = r"(?:(?:ok(?:ay)?|alright|please|hey|um|uh|well|also|ach|na|bitte)\s+)*"
_POST = r"(?:\s+(?:please|now|thanks|thank you|bitte|jetzt|danke))?"
# Time/date questions must be (nearly) the whole utterance: "what time is it in Tokyo" is not the local time.
_TAIL = r"(?:\s+(?:please|right now|now|currently|today|at the moment|bitte|gerade|jetzt))*"

_INTENTS = [
    ("stop", re.compile(rf"^{_PRE}(?:stop(?: it| that| talking| speaking)?|cancel(?: that)?|never ?mind|be quiet|quiet|shut up|enough|that'?s enough|hold on"
                        rf"|stopp?(?: das| jetzt)?|hör auf|halt|genug|das reicht|sei still|ruhe){_POST}$")),
    ("sleep", re.compile(rf"^{_PRE}(?:good ?bye|bye(?: bye)?|go to sleep|that'?s all(?: for now)?|that is all|see you(?: later)?|i'?m done|we'?re done|talk to you later|thanks?,? (?:bye|that'?s all)"
                         rf"|tschüss|tschüs|auf wiedersehen|bis später|bis bald|das war'?s|ciao){_POST}$")),
    ("repeat", re.compile(rf"^{_PRE}(?:(?:can|could|would) you )?(?:please )?(?:repeat(?: that)?|say that again|say it again|what did you say|come again|pardon(?: me)?|sorry what"
                          rf"|wiederhole(?: das)?|noch(?: ein)?mal|wie bitte|kannst du das wiederholen){_POST}$")),
    ("time", re.compile(rf"^{_PRE}(?:what(?:'s| is) the (?:current )?time|what time is it|tell me the time|do you (?:know|have) the time|current time"
                        rf"|wie spät ist es|wie viel uhr ist es|wieviel uhr ist es){_TAIL}$")),
    ("date", re.compile(rf"^{_PRE}(?:what(?:'s| is) (?:the )?(?:today'?s )?date|what day is (?:it|today)|today'?s date|what is today"
                        rf"|welcher tag ist (?:heute|es)|welches datum haben wir|der wievielte ist (?:heute|es)){_TAIL}$")),
    ("greeting", re.compile(rf"^{_PRE}(?:hello|hi|hey|hi there|hey there|hello there|good (?:morning|afternoon|evening)"
                            rf"|hallo|guten (?:morgen|tag|abend)|servus|moin)(?: there)?$")),
    ("thanks", re.compile(rf"^{_PRE}(?:thank you|thanks|thank you (?:very|so) much|thanks a (?:lot|bunch)|cheers|great thanks|perfect thanks"
                          rf"|danke|danke schön|vielen dank|dankeschön)$")),
]


# Things people say when cutting in on an answer that mean "enough, thanks": only valid while
# the assistant was talking/thinking (source barge / interject), never as a standalone request.
_CUT_IN = re.compile(rf"^{_PRE}(?:no(?: no)*|nope|wait|wait wait|okay|ok|alright|all right|got it|i got it|i see|"
                     rf"fine|yes|yeah|yep|enough|thanks|thank you|that'?s (?:fine|ok|okay|good|enough)|"
                     rf"i understand|understood|right|sure|hmm|uh huh|mm hmm"
                     rf"|nein|nee|moment|genug|ja|okay|alles klar|verstanden|gut|schon gut){_POST}$")


def classify_intent(text: str, barge: bool = False, awaiting_answer: bool = False) -> Optional[str]:
    """
    `barge`: the speech came while the assistant was talking/thinking.
    `awaiting_answer`: the assistant's last reply ended in a question, so "yes" / "okay" / "no" over it
    are the user's *answer*, not "enough" — they go to the LLM instead of stopping it.
    """
    t = normalize(text)
    if not t or len(t.split()) > 8:
        return None
    for name, pat in _INTENTS:
        if pat.search(t):
            return name
    if barge and not awaiting_answer and _CUT_IN.search(t):
        return "stop"
    return None


def ends_with_question(text: str) -> bool:
    return text.rstrip().rstrip("\"')”").endswith("?")


def answer_intent(intent: str, now: Optional[datetime] = None) -> Optional[str]:
    """Spoken text for instant intents; None for control intents (stop/sleep/repeat)."""
    now = now or datetime.now()
    if intent == "time":
        return f"It's {now.strftime('%I:%M %p').lstrip('0')}."
    if intent == "date":
        return f"Today is {now.strftime('%A, %B')} {now.day}."
    if intent == "greeting":
        return "Hello! How can I help?"
    if intent == "thanks":
        return "You're welcome!"
    return None


# ── Speech clean-up & chunking ───────────────────────────────────────────────

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]")


def clean_for_speech(text: str) -> str:
    t = text
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)       # [text](url) → text
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"[`*_~#>]+", "", t)
    t = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", t, flags=re.M)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\.(\s*\.)+", ".", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no"}
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)(\s+)")


class SentenceChunker:
    """
    Turns a stream of LLM text deltas into speakable chunks.

    * Emits on sentence boundaries (never inside "Dr.", "3.14", "e.g.").
    * The very first chunk may be released early at a clause boundary
      (comma/colon/dash) once `first_clause_words` words are buffered, so audio
      starts sooner on a slow model.
    * Tiny chunks ("Yes.") are merged with what follows to avoid choppy speech.
    """

    def __init__(self, first_clause_words: int = 12, max_words: int = 40, min_chars: int = 12):
        self.buf = ""
        self.first_clause_words = first_clause_words
        self.max_words = max_words
        self.min_chars = min_chars
        self.emitted = 0
        self._carry = ""

    def _is_abbrev(self, head: str) -> bool:
        last = re.findall(r"([\w.]+)\.+[\"')\]]*$", head.strip())
        if not last:
            return False
        w = last[0].lower().rstrip(".")
        return w in _ABBREV or (len(w) == 1 and w.isalpha())

    def _take(self, chunk: str, out: List[str]) -> None:
        chunk = clean_for_speech(self._carry + " " + chunk if self._carry else chunk)
        self._carry = ""
        if not re.search(r"\w", chunk):
            return
        if len(chunk) < self.min_chars:
            self._carry = chunk
            return
        out.append(chunk)
        self.emitted += 1

    def feed(self, delta: str) -> List[str]:
        self.buf += delta
        out: List[str] = []
        while True:
            cut = None
            for m in _BOUNDARY.finditer(self.buf):
                if self._is_abbrev(self.buf[: m.end(1)]):
                    continue
                cut = m.end()
                break
            nl = re.search(r"\n\s*\n|\n(?=\s*(?:[-•*]|\d+[.)])\s)", self.buf)
            if nl and (cut is None or nl.start() < cut):
                cut = nl.end()
            if cut is None and self.emitted == 0 and not self._carry:
                words = self.buf.split()
                if len(words) >= self.first_clause_words:
                    ms = list(re.finditer(r"[,;:—–]\s+", self.buf))
                    ms = [m for m in ms if len(self.buf[: m.end()].split()) >= 5]
                    if ms:
                        cut = ms[-1].end()
            if cut is None and len(self.buf.split()) >= self.max_words:
                sp = self.buf.rfind(" ", 0, len(self.buf) - 1)
                cut = sp + 1 if sp > 0 else None
            if cut is None:
                break
            self._take(self.buf[:cut], out)
            self.buf = self.buf[cut:]
        return out

    def flush(self) -> List[str]:
        out: List[str] = []
        rest = self.buf
        self.buf = ""
        text = (self._carry + " " + rest).strip() if self._carry else rest
        self._carry = ""
        text = clean_for_speech(text)
        if re.search(r"\w", text):
            out.append(text)
            self.emitted += 1
        return out


def trim_to_sentence(text: str) -> str:
    """Drop a trailing half-sentence (e.g. the model hit max_tokens mid-thought)."""
    t = text.strip()
    if not t or t[-1] in ".!?\"')":
        return t
    cut = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    return t[: cut + 1] if cut > len(t) * 0.4 else t + "."


def trim_for_memory(text: str, limit: int) -> str:
    """Keep the first `limit` chars of a reply but end on a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind(c, 0, limit) for c in ".!?")
    return text[: cut + 1] if cut > 0 else text[:limit]


# ── Endpointing: how patient to be after the speaker went quiet ──────────────

# A sentence cannot end on these: the speaker is mid-thought even if STT added a full stop.
DANGLING = {"and", "or", "but", "because", "the", "a", "an", "to", "of", "with", "for", "my", "your",
            "if", "then", "like", "um", "uh", "from", "by", "und", "oder", "aber", "weil", "der", "die", "das",
            "ein", "eine", "zu", "mit", "für", "wenn", "dann", "äh", "ähm"}
DANGLING_UNLESS_QUESTION = {"about", "is", "are", "was", "that", "it", "this", "there"}


def endpoint_class(raw: Optional[str], wakeword: str, wakeish: bool) -> str:
    """
    What the transcript so far says about how long to keep waiting in silence.
      "pending"     no transcript available (caller decides)
      "wake_wait"   bare wake phrase / unreadable fragment right after it → wait long for the request
      "question"    ends like a finished question or sentence → answer sooner
      "hesitation"  trails off ("…and the") → the speaker is probably still thinking
      "default"     nothing special
    The wake phrase is only looked for when the capture actually contained it (`wakeish`).
    """
    if raw is None or is_hallucination(raw):
        return "wake_wait" if wakeish else "default"
    text = clean_transcript(raw)
    core = strip_wake_phrase(text, wakeword, strict=not wakeish)
    if not core:
        return "wake_wait"
    words = core.lower().split()
    last = words[-1].strip(".,?!;:") if words else ""
    stripped = text.rstrip()
    if stripped.endswith("?") and len(words) >= 1:
        return "question"
    if stripped.endswith((",", "-", "…", "...")) or last in DANGLING:
        return "hesitation"
    if last in DANGLING_UNLESS_QUESTION and not stripped.endswith((".", "!")):
        return "hesitation"
    if stripped.endswith((".", "?", "!")) and len(words) >= 2:
        return "question"
    return "default"
