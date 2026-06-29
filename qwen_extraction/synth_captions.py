"""Generate diverse synthetic CHARACTER base-captions for dataset expansion.

These provide compositional variety (framing / setting / clothing / activity / lighting) that the
FFHQ face captions lack; the prompt_policy augment layer then injects the demographics (race, age
25-35, expression, hair, eye, makeup, jewelry) + quality/color on top. Deterministic per (batch,
index) so a re-run reproduces the same captions. Dependency-light (stdlib only).

CLI:  python synth_captions.py --count 10000 --batch 0 --out /workspace/synth_caps_0.tsv
Emits 'id<TAB>caption' lines with id = synth_b{batch}_{i:06d} (globally unique across batches and
disjoint from the numeric FFHQ ids).
"""
import argparse
import random
import zlib

FRAMING = ["A close-up portrait", "A headshot", "An upper-body portrait", "A half-body portrait",
           "A candid portrait", "A waist-up shot", "A three-quarter portrait", "A tight portrait",
           "An environmental portrait", "A medium shot", "A profile portrait", "A frontal portrait",
           "A documentary-style portrait", "A studio portrait"]
SUBJECT = ["a man", "a woman", "a man", "a woman", "a person"]   # ~40% man / 40% woman / 20% person
SETTING = ["in a cozy cafe", "in a modern office", "on a busy city street", "in a sunlit park",
           "in a home kitchen", "in a quiet library", "on a rooftop at golden hour", "in an art studio",
           "at a train station", "in a flower garden", "on a sandy beach", "in a small bookshop",
           "in a dimly lit bar", "in a glass greenhouse", "in a subway car", "against a plain studio backdrop",
           "in a forest clearing", "in a bustling market", "in a minimalist bedroom", "in a co-working space",
           "in an industrial loft", "on a snowy street", "in a vineyard", "at a rainy bus stop",
           "in a neon-lit alley", "in a sunlit living room", "on a mountain trail", "in a pottery workshop",
           "at an outdoor cafe terrace", "in a recording studio", "in a university courtyard",
           "in a tiled hallway", "by a large window", "in a barber shop", "in a tailoring atelier"]
CLOTHING = ["wearing a denim jacket", "in a wool sweater", "wearing a tailored business suit",
            "in a flowy summer dress", "wearing a hoodie", "in a leather jacket", "wearing a linen shirt",
            "in a turtleneck", "wearing a trench coat", "in a knit cardigan", "wearing a graphic t-shirt",
            "in elegant traditional attire", "wearing a raincoat", "in athletic wear", "wearing a flannel shirt",
            "in a silk blouse", "wearing dungarees", "in a puffer jacket", "wearing a blazer over a tee",
            "in a cable-knit jumper", "wearing a bomber jacket", "in a corduroy shirt", "wearing a peacoat",
            "in a tank top", "wearing a cashmere scarf and coat", "in a button-down shirt",
            "in a tweed jacket", "wearing a hospital scrubs top", "in a chef's jacket", "wearing a varsity jacket",
            "in a tailored overcoat", "wearing a polo shirt"]
ACTIVITY = ["sitting and reading a book", "holding a cup of coffee", "looking out of a window",
            "leaning against a brick wall", "standing with arms crossed", "walking down the path",
            "gazing into the distance", "resting their chin on one hand", "adjusting their collar",
            "glancing over one shoulder", "seated at a wooden desk", "standing in a doorway",
            "sketching in a notebook", "carrying a tote bag", "leaning on a railing", "tying a shoelace",
            "holding an umbrella", "glancing down thoughtfully", "stretching after a run",
            "arranging flowers", "sipping tea", "standing beside a bicycle", "looking directly at the camera",
            "turning to look back", "with their hands in their pockets", "buttoning a coat", "holding a notebook",
            "resting against a doorframe", "fixing their hair", "holding a paintbrush"]
LIGHTING = ["soft natural light", "warm golden-hour light", "moody low-key lighting", "bright midday daylight",
            "soft diffused window light", "overcast even light", "dramatic side lighting", "backlit rim light",
            "cool blue-hour light", "warm tungsten indoor light", "dappled light through leaves",
            "neon-tinted lighting", "candlelit warm glow", "soft studio lighting", "gentle overcast light"]
TEMPLATES = [
    "{framing} of {subject} {activity}, {setting}, {clothing}, {lighting}",
    "{subject}, {clothing}, {activity} {setting}, {framing_l}, {lighting}",
    "{framing} of {subject} {setting}, {clothing}, {activity}, {lighting}",
    "{framing} of {subject} {clothing}, {activity}, {setting}, with {lighting}",
    "{subject} {setting}, {activity}, {clothing}, {framing_l}, {lighting}",
]


def caption_for(batch: int, i: int) -> str:
    rng = random.Random(zlib.crc32(f"synthcap:{batch}:{i}".encode()))
    parts = dict(
        framing=rng.choice(FRAMING),
        subject=rng.choice(SUBJECT),
        setting=rng.choice(SETTING),
        clothing=rng.choice(CLOTHING),
        activity=rng.choice(ACTIVITY),
        lighting=rng.choice(LIGHTING),
    )
    parts["framing_l"] = parts["framing"][0].lower() + parts["framing"][1:]
    text = rng.choice(TEMPLATES).format(**parts)
    text = text[0].upper() + text[1:]
    return text.rstrip(". ") + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(f"synth_b{args.batch}_{i:06d}\t{caption_for(args.batch, i)}\n")
    print(f"wrote {args.count} captions for batch {args.batch} -> {args.out}")


if __name__ == "__main__":
    main()
