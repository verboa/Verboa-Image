"""Verboa Prompt Writer for ComfyUI.

One node, "Verboa Prompt Writer" (category Verboa). Connect a Load CLIP node holding the writer file
(verboa-prompt-writer-fp8.safetensors on NVIDIA, verboa-prompt-writer-bf16.safetensors on a Mac) and send its
`prompt` output to a CLIP Text Encode node's text input. Three modes:

  Let Me Choose  nine dropdowns: people, age, place, level, ethnicity, body, camera, voice, length. Leave any of them
                 on "any" and the writer invents it.
  Surprise Me    no dropdowns. Every run spins people, age, place, level, voice and length at random; the writer
                 invents the rest.
  Write My Own   a text box, passed on unchanged: the same graph renders your own prompt, and the writer file is not
                 loaded (the clip input is lazy).

The seed is set to randomize after every run, so each run writes a new prompt; fix it to keep one. The `fields`
output says what the writer was given (for Surprise Me, what was spun).

Under the hood this is ComfyUI's own text generation, called exactly as the built-in Generate Text node calls it
(thinking off, default template), with the settings the writer was evaluated at: sampling on, temperature 1.0, top_k
off, top_p off, min_p 0.05, repetition penalty 1.05, up to 768 new tokens. Needs ComfyUI 0.19.0 or newer (Qwen 3.5
text generation). Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.

A written prompt that contains a word for someone under 18 is never passed on: the node samples again (up to five
times) and stops with an error if every try has one. None of the 5,200 prompts written in our evaluation had one.
"""
from __future__ import annotations

import random
import re

from comfy_api.latest import ComfyExtension, io

FIELDS = ["people", "age", "place", "level", "ethnicity", "body", "camera", "voice", "length"]
VALUES = {
    "people": ["a woman", "a man", "two women", "two men", "a woman and a man", "a group"],
    "age": ["18-24", "25-34", "35-44", "45-54", "55+"],
    "place": ["hot tub", "shower", "bathtub", "sauna", "pool", "bathroom", "garden", "bedroom", "kitchen", "living room",
              "hotel room", "office", "gym", "beach", "forest", "balcony", "rooftop", "car", "boat", "cabin", "barn",
              "library", "studio", "stairs", "field", "lake", "desert", "mountains", "city street", "club", "dorm room",
              "laundry room", "garage", "dressing room", "massage room", "outdoors"],
    "level": ["clothed", "lingerie", "partly dressed", "nude", "explicit", "sex"],
    "ethnicity": ["East Asian", "South Asian", "Black", "Latina", "Middle Eastern", "white", "mixed"],
    "body": ["slim", "average", "athletic", "curvy", "plus-size"],
    "camera": ["pro photo", "art photo", "phone photo", "mirror selfie", "webcam"],
    "voice": ["slang", "casual", "neutral", "formal", "clinical"],
    "length": ["short", "medium", "long", "very long"],
}
ANY = "any"
# Surprise Me spins the fields the writer follows reliably. With all nine set it honored only about 5 in 8 (ethnicity
# and camera, rare in its training captions, mostly not); with these six it honored 13 to 16 of 16 per field.
SPUN = ["people", "age", "place", "level", "voice", "length"]
SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.05, "repetition_penalty": 1.05}
MAX_LENGTH = 768
TRIES = 5
LET_ME_CHOOSE, SURPRISE_ME, WRITE_MY_OWN = "Let Me Choose", "Surprise Me", "Write My Own"

# Words that put someone under 18: words for adolescents and children, school levels below college, ages under 18.
# The same list as scripts/writer/common.py MINOR (tests/test_writer_node.py checks they match). Written "t[e]en" so the
# word itself appears nowhere in the release (the user's rule, 2026-09-24); the regex matches exactly the same text.
MINOR = re.compile(
    r"\b(t[e]ens?|t[e]enage[a-z]*|t[e]enie|t[e]eny|pret[e]ens?|tweens?|child\w*|kids?|toddlers?|infants?|newborns?|minors|"
    r"minor(?!\s+(?:blemish|freckle|imperfection|detail|scar|flaw|mark|spot|variation|wrinkle|crease|bruise|redness|"
    r"irritation|asymmetr|touch|adjustment|change|shadow|reflection|highlight|stain|tear|wear|damage|bump|vein|"
    r"stretch|tan\b|discolor|texture|hair|fold|dimple|pore|mole|line)\w*)|"
    r"underage|loli\w*|shota\w*|jailbait|kindergart\w*|elementary school\w*|middle school\w*|junior high\w*|"
    r"high school\w*|(?:1[0-7]|[1-9])[ -]?(?:yo|y/o|years?[ -]old)|"
    r"(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[ -]?(?:yo|years?[ -]old))\b", re.I)


def latin_for(people: str) -> str:
    """The writer was trained on Latina for women, Latino for men, Latin for mixed groups."""
    return {"a man": "Latino", "two men": "Latino", "a woman": "Latina", "two women": "Latina"}.get(people, "Latin")


def header(fields: dict) -> str:
    """The text the writer was trained on: one `field: value` line per field, in this order."""
    return "\n".join(f"{k}: {fields.get(k) or ANY}" for k in FIELDS)


def write(clip, fields: dict, seed: int) -> str:
    for attempt in range(TRIES):
        # Generate Text's own call (0.19.0 to 0.36.0): ComfyUI adds the Qwen 3.5 chat template and an empty think block,
        # the prompt the writer was trained on
        tokens = clip.tokenize(header(fields), image=None, skip_template=False, min_length=1, thinking=False)
        ids = clip.generate(tokens, do_sample=True, max_length=MAX_LENGTH, seed=(seed + attempt) % 2**64, **SAMPLING)
        text = clip.decode(ids).strip()
        if text and not MINOR.search(text):
            return text
    raise RuntimeError(f"The Verboa Prompt Writer did not write a usable prompt in {TRIES} tries; run it again.")


class VerboaPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choose = [io.Combo.Input(k, options=[ANY] + VALUES[k], default=ANY) for k in FIELDS]
        return io.Schema(
            node_id="VerboaPromptWriter",
            display_name="Verboa Prompt Writer",
            category="Verboa",
            description="Writes a prompt for Verboa Image 1.0. Let Me Choose: set the fields you care about. "
                        "Surprise Me: every run spins them at random. Write My Own: type your own prompt instead.",
            inputs=[
                io.Clip.Input("clip", lazy=True,
                              tooltip="verboa-prompt-writer-fp8 (NVIDIA) or -bf16 (Mac), from a Load CLIP node"),
                io.DynamicCombo.Input("mode", display_name="Mode", options=[
                    io.DynamicCombo.Option(LET_ME_CHOOSE, choose),
                    io.DynamicCombo.Option(SURPRISE_ME, []),
                    io.DynamicCombo.Option(WRITE_MY_OWN, [
                        io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="")]),
                ]),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=io.ControlAfterGenerate.randomize,
                             tooltip="Randomized after every run, so each run writes a new prompt"),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="fields", tooltip="What the writer was given (for Surprise Me, the spin)"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, clip=None, mode=None, seed=None):
        choice = mode.get("mode") if isinstance(mode, dict) else None
        if isinstance(choice, tuple):                 # the lazy check gets DynamicCombo values as (value, key)
            choice = choice[0]
        return [] if choice == WRITE_MY_OWN or clip is not None else ["clip"]

    @classmethod
    def execute(cls, clip, mode, seed) -> io.NodeOutput:
        if mode.get("mode") == WRITE_MY_OWN:
            return io.NodeOutput(mode.get("prompt") or "", "Write My Own: your prompt, unchanged")
        if clip is None:
            raise RuntimeError("Connect a Load CLIP node with the Prompt Writer file to the clip input.")
        if mode.get("mode") == SURPRISE_ME:
            rng = random.Random(seed)
            fields = {k: rng.choice(VALUES[k]) if k in SPUN else ANY for k in FIELDS}
        else:
            fields = {k: mode.get(k) or ANY for k in FIELDS}
        if fields["ethnicity"] == "Latina":
            fields["ethnicity"] = latin_for(fields["people"])
        return io.NodeOutput(write(clip, fields, seed), " · ".join(f"{k}: {fields[k]}" for k in FIELDS))


class VerboaExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VerboaPromptWriter]


async def comfy_entrypoint() -> VerboaExtension:
    return VerboaExtension()
