"""Synthetic OUTFIT-caption generator to expand/rebalance the DeepFashion source.

Emits gender-gated outfit captions ("a woman wearing a <color> <garment> and <footwear>[, with
<accessory>]"). Composition (setting / pose / angle / framing / lighting / quality / wearer race +
age) is added later by prompt_policy's fashion domain, so DeepFashion-real and synthetic captions
get identical photographic treatment and only differ in the garment source. Female generation
up-weights the categories DeepFashion lacks (dresses, formal, outerwear, occasion, knitwear); the
male stream is the parallel substrate. Deterministic per (gender, batch, index). stdlib-only.

CLI: python -m qwen_extraction.fashion_captions --gender female --count 3000 --batch 0 --out f.tsv
"""
import argparse
import random
import zlib

try:
    from qwen_extraction import fashion_vocab as V
except ImportError:
    import fashion_vocab as V

# Category weights. Female up-weights the rare categories to offset DeepFashion's casual skew.
FEMALE_CAT_W = {"dresses": 20, "outerwear": 16, "formal": 14, "knitwear": 12, "occasion": 10,
                "athletic": 10, "tops": 9, "bottoms": 9}
MALE_CAT_W = {"tops": 16, "bottoms": 14, "outerwear": 14, "formal": 13, "knitwear": 11,
              "athletic": 10, "occasion": 8}
VOWELS = "aeiouAEIOU"
# footwear/accessories are shared lists in the vocab -> exclude clearly-feminine items for men
_FEM_FOOT_KW = ("heel", "pump", "slingback", "stiletto", "mary-jane", "mary jane", "mule",
                "ballet", "wedge", "gladiator", "espadrille", "evening sandal")
_FEM_ACC_KW = ("earring", "headscarf", "clutch", "purse", "handbag", "tiara", "hairpin",
               "bow", "hair clip", "scrunchie", "choker", "necklace")
_MALE_FOOTWEAR = [f for f in V.FOOTWEAR if not any(k in f.lower() for k in _FEM_FOOT_KW)]
_MALE_ACCESSORIES = [a for a in V.ACCESSORIES if not any(k in a.lower() for k in _FEM_ACC_KW)]


def _wsample(rng, weights: dict):
    items = sorted(weights.items())
    u = rng.random() * sum(w for _, w in items)
    acc = 0.0
    for k, w in items:
        acc += w
        if u < acc:
            return k
    return items[-1][0]


def _colorize(garment: str, rng: random.Random) -> str:
    """Insert a color (sometimes a patterned color) into a garment phrase ~65% of the time."""
    if rng.random() > 0.65:
        return garment
    color = rng.choice(V.COLORS)
    if rng.random() < 0.22:
        color = f"{color} {rng.choice(V.PATTERNS)}"
    if garment.startswith(("a ", "A ")):
        head, rest = garment[:2], garment[2:]
        art = "an" if color[:1] in VOWELS else "a"
        return f"{art} {color} {rest}"
    if garment.startswith(("an ", "An ")):
        rest = garment[3:]
        art = "an" if color[:1] in VOWELS else "a"
        return f"{art} {color} {rest}"
    return f"{color} {garment}"     # plural / article-less (trousers, jeans, leggings, ...)


def outfit_caption(gender: str, batch: int, idx: int) -> str:
    rng = random.Random(zlib.crc32(f"fashioncap:{gender}:{batch}:{idx}".encode()))
    if gender == "male":
        cats, subject = V.MALE_GARMENTS, "a man"
        cat = _wsample(rng, MALE_CAT_W)
        footwear_pool, acc_pool = _MALE_FOOTWEAR, _MALE_ACCESSORIES
    else:
        cats, subject = V.FEMALE_GARMENTS, "a woman"
        cat = _wsample(rng, FEMALE_CAT_W)
        footwear_pool, acc_pool = V.FOOTWEAR, V.ACCESSORIES
    garment = _colorize(rng.choice(cats[cat]), rng)
    footwear = rng.choice(footwear_pool)
    cap = f"{subject} wearing {garment} and {footwear}"
    if acc_pool and rng.random() < 0.4:
        cap += f", with {rng.choice(acc_pool)}"
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", choices=["female", "male"], required=True)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(f"fsyn_{args.gender}_b{args.batch}_{i:06d}\t{outfit_caption(args.gender, args.batch, i)}\n")
    print(f"wrote {args.count} {args.gender} outfit captions -> {args.out}")


if __name__ == "__main__":
    main()
