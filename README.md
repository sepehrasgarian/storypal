# StoryPal

A voice reading companion for children — built as a working slice of an agentic
platform: a live voice loop plus three learning timescales that let the system
adapt to the child immediately and improve itself over time.

**The design question this project answers:** when an agent's own perception is
unreliable, how should the system behave — and how does it learn from that?

## Why reading practice

Most conversational agents can't be scored objectively: there is no ground
truth for "was that a good reply?". Reading practice is different — the target
sentence is on screen, so what the child *should* have said is known exactly.
That makes the core assessment deterministic (word-level alignment) rather
than a matter of judgment.

It also sits on a known risk: children's speech is one of the hardest inputs
for speech recognition, and weak recognizers don't merely mishear — they
fabricate words that were never spoken, or silently autocorrect
mispronunciations into the right word. In a reading tutor both failures are
directly harmful: the system corrects a child who was right, or praises a
mistake. This architecture treats recognition confidence as a first-class
signal for exactly that reason.

## Architecture

```
BROWSER                      API (FastAPI)                          EXTERNAL
────────                     ─────────────                          ────────
🎤 record ── POST /api/turn ──► asr.py ──────────────────────────►  Whisper (local)
                                  │  transcript + confidence
                                  │  telemetry
                                  ▼
                    ┌── SYNC PRE-LOOP (deterministic, free) ──┐
                    │  assessment.py  normalized fuzzy         │
                    │                 alignment vs target      │
                    │  S1 reading accuracy                     │
                    │  S2 ASR reliability  ◄── gates S1        │
                    └──────────────────┬───────────────────────┘
                                       ▼
                                  prompt.py   (Tier 1: signals + profile)
                                       ▼
                                  agent.py + tools.py ───────►  LLM provider
                                       ▼
                                  tts.py  ───────────────────►  Higgs TTS v3
                                       │                        (Boson API)
🔊 play ◄── JSON reply ────────────────┘
                                       │
                    ┌── ASYNC POST-LOOP (after reply sent) ────┐
                    │  S3 tutor grounding   (LLM judge)        │
                    │  S4 pedagogical fit   (LLM judge)        │
                    │       │                                  │
                    │       ▼                                  │
                    │  triage.py ──► Tier 3 curated data       │
                    │       └──────► next turn's Tier 1 prompt │
                    └───────────────────────────────────────────┘
```

The deterministic signals (S1, S2) run synchronously inside the turn; the LLM
judges (S3, S4) run asynchronously after the reply is sent, so quality
judgment never sits on the latency path.

## The three learning timescales

| Tier | Mechanism | Latency | Cost | Persisted as |
|------|-----------|---------|------|--------------|
| 1 | Feedback injected into the prompt | next turn | free | nothing (ephemeral) |
| 2 | Learner profile loaded into the prompt | next session | free | `data/profile.json` |
| 3 | Curated failures for offline fine-tuning | days | GPUs | `data/curated/*.jsonl` |

**Tier 1** rebuilds the prompt every turn from live signals. When the
recognizer is judged unreliable, the tutor's instructions flip from "correct
the child" to "ask them to read it once more" — perception quality is routed
directly into the agent's behaviour.

**Tier 2** is a small learner profile (weak phonemes, missed words, pace,
difficulty) accumulated across sessions. Turns flagged unreliable by S2 are
excluded from profile updates, so recognition artifacts never poison memory.

**Tier 3** produces the dataset a training run would consume: each curated
record stores the full context, the flawed reply, the judge's verdict, and a
slot for a corrected reply — yielding SFT/DPO-ready pairs. No training is run
here; the point is the curation judgment.

## Signals

| ID | Signal | Method | Detects |
|----|--------|--------|---------|
| S1 | Reading accuracy | word alignment vs target | missed / substituted / added words |
| S2 | ASR reliability | ASR telemetry + target-anchored novelty | recognizer fabricating content |
| S3 | Tutor grounding | LLM judge | tutor claims the assessment doesn't support |
| S4 | Pedagogical fit | LLM judge vs rubric | corrections that are wrong, harsh, or off-target |

S2 is the safeguard: it gates what the tutor says now (Tier 1), what the
system remembers later (Tier 2), and what data reaches training (Tier 3).

## Voice output

Delivery is chosen by the signals rather than fixed, using Higgs TTS v3's
expressive tags: warm praise after a strong reading, an encouraging tone for
gentle corrections, slowed prosody when modelling a difficult word.

## Project layout

```
src/storypal/
  config.py     every tunable number in one place
  core/         assessment, signals, triage, trajectory   (deterministic heart)
  learning/     profile, kb, prompt, curated              (the three tiers)
  agent/        llm, loop, tools, judges                  (the agentic layer)
  speech/       asr, tts                                  (ears and voice)
  api/          FastAPI wiring
web/            single page, no build step
eval/           labeled cases + metrics report
tests/          one test file per module
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
brew install ffmpeg          # Whisper needs it to decode browser audio
cp .env.example .env         # then fill in your API keys
uvicorn storypal.api.main:app --reload
```

Run tests and the eval report:

```bash
pytest
python -m eval.run_eval
```

## Deliberate limitations

- No training is run; Tier 3 produces the dataset only.
- S2 catches fabrication but not autocorrection (a strong-LM recognizer
  "fixing" a child's mispronunciation into the right word). The real fix is a
  phoneme-level acoustic model.
- Turn-taking is push-to-talk rather than streaming; barge-in needs a live
  speech-to-speech runtime.
- Judge-based signals (S3, S4) are unvalidated against human ratings.
- The learner profile is a single-child JSON file, not a multi-tenant store.

## Status

Early development — deterministic core (assessment, signals, triage) first,
then the agent loop, then voice in/out.
