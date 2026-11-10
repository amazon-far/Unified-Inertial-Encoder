"""LLM transcription of activity descriptions into IMU-observable motion captions."""
import os
import boto3
import anthropic
import botocore.exceptions
import typing as tp

T_Backend: tp.TypeAlias = tp.Literal["claude-haiku-4-5"]

_AWS_REGION_ = "us-west-2"
_MODEL_ID_: dict[T_Backend, str] = {
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Keeps only what whole-body IMU and pose can observe, as a short natural phrase with every
# motion in order. `cache_control` caches the prompt across the dataset sweep.
_SYSTEM_PROMPT_ = """\
You convert third-person activity descriptions from a motion-capture dataset into minimal \
motion transcriptions. The transcription will caption body-worn IMU (inertial) recordings, so \
it must contain ONLY what is observable from the wearer's whole-body movement and pose over time.

Rules:
- Keep locomotion and posture: walking, standing, running, sitting, bending down, turning, \
climbing stairs, descending stairs, ...
- Keep manipulation as generic arm/hand motion: picking up something, placing something, \
carrying something, folding something, reaching, opening, closing, ...
- Remove everything not observable from body motion alone: object identity (bag, clothes, \
glass), rooms and locations (kitchen, living area), other people, speech ("talking to her \
peer"), intentions and instructions ("as instructed by...").
- Preserve every distinct motion and their temporal order. Do not condense to a single motion.
- Merge consecutive repetitions of the same motion into one mention.
- Output one short natural phrase in present continuous tense with no subject, e.g. \
"folding something and placing it into a container while standing". Connect stages with \
"then" when the person transitions between activities.
- Output ONLY the phrase: no quotes, no explanation, no leading/trailing whitespace.

Examples:

Input: C is folding the clothes and putting them into the eco bag in the living area while \
talking with her peer.
Output: standing while folding something and placing it into a container

Input: C is walking towards the kitchen opens the cabinet and puts the plastic on the trash \
bin and then walks towards the dining area as she grab the box of the board game
Output: walking, then opening something and placing an object, then walking and picking \
something up

Input: C exits the bedroom, descends the stairs, and proceeds to the activity area, where \
she stands as advised by her peer.
Output: walking, then descending stairs, then walking and standing

Input: C is talking to her peer in the dining area
Output: standing\
"""

# Same observability constraints in the BONES-SEED temporal-label voice, so the Nymeria and
# BONES captions share one CLIP text distribution.
_BONES_STYLE_PROMPT_ = """\
You convert third-person activity descriptions from a motion-capture dataset into motion \
captions written in the style of motion-capture temporal labels. The caption will label \
body-worn IMU (inertial) recordings, so it must contain ONLY what is observable from the \
wearer's whole-body movement and pose over time.

Rules:
- Write ONE sentence in simple present tense with the subject "A person": \
"A person walks forward, then bends down and picks up an object."
- Keep locomotion and posture: walking, standing, running, sitting, bending down, turning, \
climbing stairs, descending stairs, ...
- Keep manipulation as generic limb motion on generic objects: picks up an object, places \
an object, carries an object, folds an object, reaches, opens a cabinet, ... Use "an \
object" / "a container" / "a cabinet"; never name the real item (bag, clothes, glass).
- Keep body detail the source states: which hand/arm/leg, both hands, posture (standing, \
seated, crouching), and movement direction.
- Remove everything not observable from body motion alone: rooms and locations, other \
people, speech, names, intentions and instructions.
- Preserve every distinct motion and their temporal order, connecting stages with "then". \
Do not condense to a single motion. Merge consecutive repetitions of the same motion into \
one mention.
- Output ONLY the sentence: no quotes, no explanation, no leading/trailing whitespace.

Examples:

Input: C is folding the clothes and putting them into the eco bag in the living area while \
talking with her peer.
Output: A person standing upright folds an object with both hands and places it into a container.

Input: C is walking towards the kitchen opens the cabinet and puts the plastic on the trash \
bin and then walks towards the dining area as she grab the box of the board game
Output: A person walks forward and opens a cabinet to place an object, then walks and picks \
up an object.

Input: C exits the bedroom, descends the stairs, and proceeds to the activity area, where \
she stands as advised by her peer.
Output: A person walks forward, descends the stairs, then walks and stands still.

Input: C reaches for the cup on the table with her right hand and drinks while seated on the couch.
Output: A person sitting reaches forward with their right hand and raises it to their mouth.

Input: C is talking to her peer in the dining area
Output: A person is standing still.\
"""

_client: anthropic.AnthropicBedrock | None = None


def _get_client() -> anthropic.AnthropicBedrock:
    """Lazily construct and cache the Bedrock client.

    An `AWS_PROFILE` naming a missing profile is dropped from the environment so boto3 falls through
    to instance-role credentials; the SDK reads the variable at request-signing time.
    """
    global _client
    if _client is None:
        try:
            boto3.Session().get_credentials()
        except botocore.exceptions.ProfileNotFound:
            os.environ.pop("AWS_PROFILE", None)
        _client = anthropic.AnthropicBedrock(aws_region=_AWS_REGION_)
    return _client


def _transcribe(system_prompt: str, raw_description: str, backend: T_Backend) -> str:
    response = _get_client().messages.create(
        model=_MODEL_ID_[backend],
        max_tokens=256,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": raw_description.strip()}],
    )

    assert response.stop_reason == "end_turn", f"unexpected stop: {response.stop_reason}"
    block = response.content[0]
    assert isinstance(block, anthropic.types.TextBlock)
    return block.text.strip()


def transcribe_extract_motion_verb(raw_description: str, backend: T_Backend="claude-haiku-4-5") -> str:
    """Condense a Nymeria activity description to a minimal IMU-observable motion phrase."""
    return _transcribe(_SYSTEM_PROMPT_, raw_description, backend)


def transcribe_bones_style(raw_description: str, backend: T_Backend="claude-haiku-4-5") -> str:
    """Rewrite a Nymeria activity description as a BONES-SEED-style "A person ..." sentence."""
    return _transcribe(_BONES_STYLE_PROMPT_, raw_description, backend)
