"""Session-level evaluation: run whole sessions with simulated children
and measure whether the system's beliefs converge on the truth.

Single-turn evals ask "was this graded correctly?". This asks the
questions that only appear over a session:

  false corrections  - did we correct a child who read it perfectly?
                       (the harm the whole architecture exists to avoid)
  false praise       - did we accept a read that was actually wrong?
  detection latency  - how many turns to identify a real weakness?
  human load         - what share of turns got punted to review?

Usage:  python -m eval.simulate
"""

import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eval.personas import PERSONAS, Persona
from storypal.config import STORIES
from storypal.core.assessment import assess, normalize
from storypal.core.triage import Route, route_turn
from storypal.learning.kb import TacticStats, next_sentence
from storypal.learning.profile import Profile, update_from_turn, weakest_phoneme
from storypal.session import grade_turn, update_expectations

TURNS = 24


@dataclass
class Outcome:
    persona: str
    turns: int = 0
    graded: int = 0
    discarded: int = 0  # S2 refused to trust the transcript
    chat: int = 0
    false_corrections: int = 0  # read perfectly, but we acted on a "mistake"
    false_praise: int = 0  # read wrongly, but we accepted it
    advances: int = 0
    detected_at: int | None = None  # turn the true weakness surfaced
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        rate = lambda n: f"{n / self.turns:.0%}" if self.turns else "-"  # noqa: E731
        detect = f"turn {self.detected_at}" if self.detected_at else "-"
        return (f"{self.persona:18} graded {rate(self.graded):>4}  "
                f"discarded {rate(self.discarded):>4}  chat {rate(self.chat):>4}  "
                f"false-correct {self.false_corrections:>2}  "
                f"false-praise {self.false_praise:>2}  "
                f"advanced {self.advances:>2}  detect {detect}")


def run_session(persona: Persona, seed: int = 7, turns: int = TURNS) -> Outcome:
    rng = random.Random(seed)
    stats = TacticStats(Path(tempfile.mkdtemp()) / "tactics.json")
    profile = Profile(level=1)
    outcome = Outcome(persona=persona.name)
    target = STORIES[0].text
    seen, pending, tactic = {target}, None, None

    for turn in range(1, turns + 1):
        utterance = persona.speak(target, rng)
        graded = grade_turn(target, utterance.transcript, utterance.telemetry, pending)
        outcome.turns += 1

        read_perfectly = normalize(utterance.truth) == normalize(target)

        if graded.chat_turn:
            outcome.chat += 1
        elif not graded.s2.reliable:
            outcome.discarded += 1
        else:
            outcome.graded += 1
            update_from_turn(profile, graded.assessment, graded.s2)
            # The failure that matters: the child said it right, we did not
            # believe them, and we acted on that belief.
            if read_perfectly and graded.s1.score < 1.0:
                outcome.false_corrections += 1
            if not read_perfectly and graded.accepted:
                outcome.false_praise += 1

        if (outcome.detected_at is None and persona.weakness
                and weakest_phoneme(profile) == persona.weakness):
            outcome.detected_at = turn

        pending, tactic = update_expectations(graded, tactic, pending, tactic)
        if graded.accepted:
            outcome.advances += 1
            story = next_sentence(profile.level, weakest_phoneme(profile), exclude=seen)
            if story is None:
                seen = set()
                story = next_sentence(profile.level, weakest_phoneme(profile))
            target = story.text
            seen.add(target)

    if persona.weakness and outcome.detected_at is None:
        outcome.notes.append(f"never identified the real weakness /{persona.weakness}/")
    return outcome


def main() -> int:
    print("# StoryPal session simulation\n")
    print(f"{TURNS} turns per persona, seeded and deterministic.\n")
    outcomes = [run_session(p) for p in PERSONAS]
    for o in outcomes:
        print("  " + o.summary)

    print("\nWhat the numbers must show:")
    failures = []
    for o in outcomes:
        if o.false_corrections:
            failures.append(f"{o.persona}: {o.false_corrections} false correction(s)")
        for note in o.notes:
            failures.append(f"{o.persona}: {note}")

    mumbler = next(o for o in outcomes if o.persona == "tired_mumbler")
    print(f"  - a misheard child is protected: {mumbler.discarded}/{mumbler.turns} "
          f"turns discarded, {mumbler.false_corrections} false corrections")
    if mumbler.advances == 0:
        print("    ...but also completely stuck: 0 sentences advanced. Caution has a")
        print("    price, and the fix is a better acoustic model, not a looser threshold.")
    struggler = next(o for o in outcomes if o.persona == "th_struggler")
    print(f"  - a real weakness is found: /th/ detected at turn {struggler.detected_at}")
    chatty = next(o for o in outcomes if o.persona == "chatterbox")
    print(f"  - conversation is not graded: {chatty.chat}/{chatty.turns} turns answered, not scored")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll session-level expectations met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
