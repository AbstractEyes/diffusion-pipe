"""Prompt augmentation / "subtext structure" policy for Qwen-Image synthetic-character generation.

Two problems this solves:
1. RACE: Qwen-Image collapses toward asian-looking faces when race is unspecified -> we inject a
   fair, configurable race mix (respecting any race the source states).
2. ATTRIBUTE FLOODING: the FFHQ llava source captions say "smiling" ~67% of the time (and the
   model defaults to smile/neutral), and rarely specify makeup/eye-color/jewelry -> the raw set
   is monotonous. We STRIP the source expression and inject a diverse weighted set of expressions,
   and inject hair-color / eye-color / makeup (women) / jewelry when unspecified. We also stamp a
   quality tier (mostly realistic color photography, ~1/10 amateur) and ENFORCE color (never
   monochrome/greyscale) and a 25-35 adult age.

Everything is deterministic per row id (blake2b seed). Explicit source attributes are respected;
only missing ones are injected. Expression is the exception: it is OVERRIDDEN (source smile
stripped) because ~2/3 of sources smile and we want a broad emotional spread.

A separate strong age-verification filter runs LATER (not here). Dependency-light: stdlib only.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

POLICY_VERSION = "augment-v2"


# ======================================================================================
# RACE taxonomy (unchanged distribution: more-uniform, Caucasian slightly highest, rare tail)
# ======================================================================================
RACE_DESCRIPTORS = {
    "caucasian": "Caucasian", "east_asian": "East Asian", "south_asian": "South Asian",
    "southeast_asian": "Southeast Asian", "black": "Black", "hispanic_latino": "Hispanic",
    "middle_eastern": "Middle Eastern", "native_american": "Native American",
    "pacific_islander": "Pacific Islander", "multiracial": "mixed-race",
    "central_asian": "Central Asian", "mediterranean": "Mediterranean", "persian": "Persian",
    "ethiopian": "Ethiopian", "scandinavian": "Scandinavian", "polynesian": "Polynesian",
    "mestizo": "Mestizo",
}
TOP10_WEIGHTS = {"caucasian": 15, "east_asian": 8, "black": 8, "south_asian": 8,
                 "hispanic_latino": 8, "middle_eastern": 8, "southeast_asian": 8,
                 "native_american": 8, "pacific_islander": 7, "multiracial": 7}
TAIL_WEIGHTS = {"central_asian": 3, "mediterranean": 2, "persian": 2, "ethiopian": 2,
                "scandinavian": 2, "polynesian": 2, "mestizo": 2}
RACE_PHRASES = {
    "caucasian": ["caucasian", "white", "european", "anglo", "nordic", "slavic"],
    "east_asian": ["east asian", "asian", "chinese", "japanese", "korean", "taiwanese", "mongolian"],
    "southeast_asian": ["southeast asian", "south-east asian", "filipino", "filipina", "vietnamese",
                        "thai", "indonesian", "malay", "cambodian", "burmese"],
    "south_asian": ["south asian", "indian", "pakistani", "bangladeshi", "sri lankan", "nepali", "desi"],
    "black": ["black", "african american", "african-american", "african", "afro-caribbean",
              "sub-saharan", "nigerian", "ghanaian"],
    "hispanic_latino": ["hispanic", "latino", "latina", "latinx", "mexican", "brazilian",
                        "colombian", "cuban", "puerto rican"],
    "middle_eastern": ["middle eastern", "middle-eastern", "arab", "arabic", "turkish",
                       "lebanese", "egyptian"],
    "native_american": ["native american", "indigenous", "first nations", "american indian", "amerindian"],
    "pacific_islander": ["pacific islander", "hawaiian", "samoan", "maori", "melanesian", "micronesian"],
    "multiracial": ["multiracial", "mixed race", "mixed-race", "biracial", "mixed descent"],
    "central_asian": ["central asian", "kazakh", "uzbek", "afghan"],
    "mediterranean": ["mediterranean", "greek", "italian", "spanish", "portuguese"],
    "persian": ["persian", "iranian"], "ethiopian": ["ethiopian", "eritrean", "somali"],
    "scandinavian": ["scandinavian", "swedish", "norwegian", "danish", "finnish", "icelandic"],
    "polynesian": ["polynesian", "tongan", "fijian"], "mestizo": ["mestizo", "mestiza"],
}
# Races whose hair/eyes skew dark (lightly realistic conditioning; still allows variety).
DARK_FEATURE_RACES = {"east_asian", "southeast_asian", "south_asian", "black", "native_american",
                      "pacific_islander", "middle_eastern", "persian", "ethiopian", "mestizo",
                      "central_asian", "polynesian"}

# ======================================================================================
# ATTRIBUTE taxonomies (clauses are ready to drop in after a comma)
# ======================================================================================
# Expressions: smile/neutral are deliberately a MINORITY; broad emotional spread otherwise.
EXPRESSIONS = {
    "with a neutral expression": 9, "with a soft smile": 7, "with a faint smile": 5,
    "with a warm smile": 5, "smiling gently": 4, "laughing": 3, "with a serious expression": 8,
    "with a stern expression": 5, "with a thoughtful expression": 7, "with a pensive look": 6,
    "with a contemplative gaze": 5, "with a calm expression": 6, "with a confident expression": 6,
    "with a determined look": 5, "with a curious expression": 5, "with a surprised expression": 4,
    "with a melancholic expression": 4, "with a wistful look": 4, "with a focused gaze": 5,
    "with a playful expression": 4, "with a skeptical look": 4, "with a somber expression": 4,
    "with a serene expression": 5, "with raised eyebrows": 3, "looking off to the side": 4,
    "gazing directly at the camera": 5, "with a slightly furrowed brow": 4, "with parted lips": 3,
    "with a subtle frown": 4, "with an intense gaze": 4,
}
HAIR_DARK = {"black hair": 26, "dark brown hair": 26, "brown hair": 16, "jet black hair": 8,
             "espresso brown hair": 5, "dark hair with subtle highlights": 5,
             "brown hair with caramel highlights": 4, "dyed auburn hair": 3, "dyed burgundy hair": 2}
HAIR_FULL = {"black hair": 8, "dark brown hair": 14, "brown hair": 12, "light brown hair": 9,
             "chestnut hair": 6, "auburn hair": 6, "red hair": 4, "copper hair": 3,
             "strawberry blonde hair": 4, "blonde hair": 11, "honey blonde hair": 4,
             "platinum blonde hair": 3, "dark blonde hair": 6, "ash brown hair": 4,
             "gray hair": 2, "silver hair": 1}
EYE_DARK = {"brown eyes": 32, "dark brown eyes": 28, "hazel eyes": 12, "amber eyes": 8,
            "near-black eyes": 8, "green eyes": 4, "gray eyes": 4}
EYE_FULL = {"brown eyes": 20, "dark brown eyes": 13, "hazel eyes": 12, "amber eyes": 5,
            "green eyes": 14, "blue eyes": 16, "gray eyes": 8, "blue-gray eyes": 5,
            "light brown eyes": 4}
MAKEUP = {  # women only
    "with no visible makeup": 16, "with natural makeup": 13, "with light makeup": 11,
    "with minimal makeup": 9, "with subtle everyday makeup": 7, "with soft glam makeup": 5,
    "wearing bold red lipstick": 5, "wearing pink lipstick": 4, "wearing nude lipstick": 4,
    "with smoky eye makeup": 5, "with winged eyeliner": 4, "with rosy blush": 3,
    "with dewy skin and natural makeup": 3, "with matte makeup": 3,
    "with glamorous evening makeup": 3, "with bold eye makeup": 3, "with neutral-toned makeup": 4,
}
JEWELRY_WOMEN = {
    "wearing no jewelry": 28, "with small stud earrings": 9, "with hoop earrings": 7,
    "wearing a delicate necklace": 7, "with a pendant necklace": 5, "with drop earrings": 4,
    "wearing a thin gold chain": 4, "with a silver necklace": 4, "wearing a single ring": 3,
    "with minimal jewelry": 6, "with statement earrings": 3, "wearing a choker": 3,
    "with layered necklaces": 3, "wearing pearl earrings": 3,
}
JEWELRY_MEN = {"wearing no jewelry": 58, "with a thin chain necklace": 8, "with a single stud earring": 6,
               "wearing a simple ring": 6, "with a wristwatch": 12, "with minimal jewelry": 10}
# Quality tiers -- every phrase contains "color" to enforce a color photograph (never monochrome).
QUALITY_PHOTO = {
    "High-quality realistic color portrait photograph with natural skin texture": 10,
    "Professional color portrait photograph, sharp focus, natural lighting": 9,
    "Realistic color DSLR portrait, shallow depth of field": 8,
    "Studio color portrait photograph with soft lighting": 7,
    "Crisp color headshot photograph, true-to-life skin tones": 7,
    "Photorealistic color portrait, detailed skin and hair": 8,
    "Natural-light color photograph, candid and realistic": 6,
    "Color portrait photograph with balanced lighting and fine detail": 7,
}
QUALITY_AMATEUR = {  # ~1/10 -- slightly reduced quality, still realistic color
    "Amateur color snapshot, slightly soft focus, casual lighting": 5,
    "Candid color phone photo, everyday lighting, slightly grainy": 5,
    "Casual amateur color photograph, imperfect framing": 4,
    "Everyday color snapshot, simple point-and-shoot look, mild noise": 4,
}

# ======================================================================================
# Detection vocab
# ======================================================================================
WOMAN_TOKENS = ["woman", "women", "female", "lady", "girl", "girls", "mother", "wife", "she"]
MAN_TOKENS = ["man", "men", "male", "gentleman", "guy", "boy", "boys", "father", "husband"]
MINOR_GENDER_TOKENS = {"girl", "girls", "boy", "boys"}
# explicit hair/eye/makeup/jewelry surfaces -> "respect if present"
HAIR_SURFACES = ["blonde", "blond", "brunette", "auburn", "redhead", "red-haired", "ginger",
                 "brown hair", "black hair", "dark hair", "light hair", "red hair", "gray hair",
                 "grey hair", "white hair", "silver hair", "platinum", "chestnut", "copper"]
EYE_SURFACE_RE = re.compile(r"\b(blue|brown|green|hazel|gray|grey|amber|dark|light|black)[\s-]+eyes\b", re.I)
MAKEUP_SURFACE_RE = re.compile(r"\b(makeup|make-up|lipstick|eyeshadow|mascara|eyeliner|blush|lip\s+gloss|foundation)\b", re.I)
JEWELRY_SURFACE_RE = re.compile(r"\b(earrings?|necklace|jewelry|jewellery|pendant|bracelet|choker|brooch)\b", re.I)
EXPRESSION_PRESENT_RE = re.compile(r"\b(smil\w*|grin\w*|laugh\w*|frown\w*|neutral expression|serious expression)\b", re.I)

AGE_MINOR_TERMS = ["youthful", "young", "teenage", "teenager", "teen", "adolescent", "children",
                   "child", "kids", "kid", "baby", "infant", "toddler", "juvenile", "minor",
                   "schoolgirl", "schoolboy", "little girl", "little boy", "underage", "preteen"]
AGE_OLD_TERMS = ["elderly", "middle-aged", "middle aged", "geriatric", "senior", "wrinkled",
                 "gray-haired", "grey-haired", "white-haired", "grandmother", "grandfather",
                 "grandma", "grandpa", "aged", "old"]
AGE_NUMERIC_PATTERNS = [
    r"\b(?:likely\s+)?(?:in\s+)?(?:his|her|their)\s+(?:late\s+|early\s+|mid[\s-]*)?\d{1,2}s(?:\s+to\s+(?:late\s+|early\s+|mid[\s-]*)?\d{1,2}s)?\b",
    r"\b\d{1,2}\s*[-\s]?\s*years?[\s-]?old\b",
    r"\baged?\s+\d{1,2}\b",
    r"\b(?:late|early|mid)[\s-]*\d{1,2}s\b",
    r"\bin\s+(?:his|her|their)\s+(?:teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
]
# expression phrases to STRIP from the source (longest-first applied at runtime)
EXPRESSION_STRIP = [
    "a wide smile", "a bright smile", "a warm smile", "a slight smile", "a soft smile",
    "a big smile", "a gentle smile", "a subtle smile", "a small smile", "a friendly smile",
    "a beaming smile", "a broad smile", "a wide grin", "a slight grin", "an expressionless face",
    "a neutral expression", "a serious expression", "a confident expression", "a playful expression",
    "a stern expression", "a thoughtful expression", "grinning from ear to ear",
    "smiling from ear to ear", "smiling broadly", "smiling warmly", "smiling gently",
    "smiling slightly", "grinning broadly", "grinning widely", "laughing heartily",
    "a smile", "smiling", "smiles", "smile", "grinning", "a grin", "grins", "laughing",
    "smirking", "beaming", "frowning", "a frown",
]
BW_STRIP = ["black and white", "black-and-white", "monochrome", "monochromatic", "greyscale",
            "grayscale", "sepia", "b&w", "in b and w"]


@dataclass
class PromptAugmentConfig:
    top10_weights: dict = field(default_factory=lambda: dict(TOP10_WEIGHTS))
    tail_weights: dict = field(default_factory=lambda: dict(TAIL_WEIGHTS))
    default_race: str = "caucasian"
    age_band: str = "25-35"
    inject_only_if_person: bool = True
    asian_generic_to: str = "east_asian"
    amateur_fraction: float = 0.1       # ~1 in 10 images get a slightly-reduced amateur quality tag
    inject_expression: bool = True
    inject_hair: bool = True
    inject_eye: bool = True
    inject_makeup: bool = True
    inject_jewelry: bool = True
    seed_salt: str = "qwen-synth-v2"


# ======================================================================================
# Helpers
# ======================================================================================
def stable_seed(row_id, salt: str = "qwen-synth-v2") -> int:
    h = hashlib.blake2b(f"{salt}:{row_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def _word_re(phrase: str) -> re.Pattern:
    esc = re.escape(phrase).replace(r"\ ", r"[\s-]+")
    return re.compile(rf"\b{esc}\b", re.IGNORECASE)


_PHRASE_LABELS = []
for _label, _phrases in RACE_PHRASES.items():
    for _p in _phrases:
        _PHRASE_LABELS.append((_p, _label))
_PHRASE_LABELS.sort(key=lambda pl: -len(pl[0]))
_PHRASE_RE = {p: _word_re(p) for p, _ in _PHRASE_LABELS}
_COLOR_RACE_RE = {
    "white": re.compile(r"\b(white\s+(?:man|men|woman|women|male|female|person|people|lady|gentleman|guy|individual|skin|complexion)"
                        r"|(?:man|woman|male|female|person|skin|complexion)\s+(?:who\s+is\s+|is\s+)?white)\b", re.IGNORECASE),
    "black": re.compile(r"\b(black\s+(?:man|men|woman|women|male|female|person|people|lady|gentleman|guy|individual|skin|complexion)"
                        r"|(?:man|woman|male|female|person|skin|complexion)\s+(?:who\s+is\s+|is\s+)?black)\b", re.IGNORECASE),
}


def detect_race(text: str, cfg: PromptAugmentConfig):
    m = re.search(r"\bof\s+([a-z\- ]+?)\s+descent\b", text, re.IGNORECASE)
    if m:
        inner = m.group(1).strip().lower()
        for phrase, label in _PHRASE_LABELS:
            if phrase in inner:
                return _normalize_label(label, cfg), m.group(0)
    for phrase, label in _PHRASE_LABELS:
        if phrase in ("white", "black"):
            mm = _COLOR_RACE_RE[phrase].search(text)
            if mm:
                return _normalize_label(label, cfg), mm.group(0)
            continue
        mm = _PHRASE_RE[phrase].search(text)
        if mm:
            return _normalize_label(label, cfg), mm.group(0)
    return None, None


def _normalize_label(label, cfg):
    return cfg.asian_generic_to if label == "east_asian" else label


def detect_gender(text: str):
    best = None
    for token, gender in [(t, "woman") for t in WOMAN_TOKENS] + [(t, "man") for t in MAN_TOKENS]:
        m = _word_re(token).search(text)
        if m:
            cand = (m.start(), gender, token in MINOR_GENDER_TOKENS)
            if best is None or cand[0] < best[0]:
                best = cand
    return ("person", False) if best is None else (best[1], best[2])


def detect_hair(text: str):
    for color in sorted(HAIR_SURFACES, key=lambda c: -len(c)):
        if _word_re(color).search(text):
            return color
    return None


def has_eye(text):
    return EYE_SURFACE_RE.search(text) is not None


def has_makeup(text):
    return MAKEUP_SURFACE_RE.search(text) is not None


def has_jewelry(text):
    return JEWELRY_SURFACE_RE.search(text) is not None


def strip_age_terms(text: str):
    removed, out = [], text
    for pat in AGE_NUMERIC_PATTERNS:
        for m in re.finditer(pat, out, re.IGNORECASE):
            removed.append(m.group(0))
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\blikely\b|\bappears?\s+to\s+be\b|\bappearing\s+to\s+be\b", "", out, flags=re.IGNORECASE)
    for term in sorted(AGE_MINOR_TERMS + AGE_OLD_TERMS, key=lambda t: -len(t)):
        rx = _word_re(term)
        if rx.search(out):
            removed.append(term)
            out = rx.sub("", out)
    return _tidy(out), removed


def strip_expression(text: str):
    out = text
    for ph in EXPRESSION_STRIP:
        out = _word_re(ph).sub("", out)
    # repair the common "<smile> on her face/lips" carrier left orphaned by the strip above
    out = re.sub(r"\bwith\s+(?:a\s+)?on\s+(?:his|her|their)\s+(?:face|lips|features?)\b", "", out, flags=re.I)
    out = re.sub(r"(?:,|\band\b)\s*on\s+(?:his|her|their)\s+(?:face|lips)\b", "", out, flags=re.I)
    return _tidy(out)


def strip_bw(text: str):
    out = text
    for ph in BW_STRIP:
        out = _word_re(ph).sub("", out)
    return _tidy(out)


_DEMINOR = [(re.compile(r"\blittle\s+girls?\b", re.I), "woman"), (re.compile(r"\blittle\s+boys?\b", re.I), "man"),
            (re.compile(r"\bschoolgirls?\b", re.I), "woman"), (re.compile(r"\bschoolboys?\b", re.I), "man"),
            (re.compile(r"\bgirls\b", re.I), "women"), (re.compile(r"\bboys\b", re.I), "men"),
            (re.compile(r"\bgirl\b", re.I), "woman"), (re.compile(r"\bboy\b", re.I), "man")]


def deminor(text: str) -> str:
    for rx, repl in _DEMINOR:
        text = rx.sub(repl, text)
    return text


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # drop dangling connectors left by removals, before/after punctuation (with or without space):
    # "round face and , wearing" -> "round face, wearing"; "beard, broadly" -> "beard"
    text = re.sub(r"\b(a|an|the|with|wearing|has|having|featuring|and)\s*(?=[,.;])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[,;])\s*(and|with|has|having)\s+(?=[,.;])", " ", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*(broadly|widely|warmly|heartily|cheerfully|from ear to ear)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,;:])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\ban\s+(?=[^aeiouAEIOU\s])", "a ", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s+(?=[aeiouAEIOU])", "an ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip(" ,;.")


def _age_phrase(gender):
    return {"woman": "in her late twenties to early thirties",
            "man": "in his late twenties to early thirties"}.get(gender, "in their late twenties to early thirties")


def _race_descriptor(label, gender):
    if label == "hispanic_latino":
        return {"woman": "Latina", "man": "Latino"}.get(gender, "Hispanic")
    return RACE_DESCRIPTORS[label]


def _article(first_word):
    return "an" if first_word[:1].lower() in "aeiou" else "a"


def sample_weighted(rng, weights: dict):
    items = sorted(weights.items())          # fixed order for determinism
    total = sum(w for _, w in items)
    u = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if u < acc:
            return k
    return items[-1][0]


def sample_race(rng, cfg):
    combined = {**cfg.top10_weights, **cfg.tail_weights}
    label = sample_weighted(rng, combined)
    return label, (label in cfg.tail_weights)


_SUBJECT_RE = re.compile(
    r"\b(a|an|the)\s+((?:[a-z]+\s+){0,3}?)(men|man|women|woman|male|female|people|person|lady|gentleman|guy|individual)\b",
    re.IGNORECASE)
_ADULT_NOUN = {"male": "man", "female": "woman", "lady": "woman", "gentleman": "man",
               "guy": "man", "individual": "person"}


def _inject_race_into_subject(text, descriptor):
    m = _SUBJECT_RE.search(text)
    if not m:
        return text, False
    article, adjs, noun = m.group(1), m.group(2) or "", m.group(3)
    adult = _ADULT_NOUN.get(noun.lower(), noun)
    return text[:m.start()] + f"{article} {adjs}{descriptor} {adult}" + text[m.end():], True


_PERSON_TOKEN_RE = re.compile(
    r"\b(man|men|woman|women|male|female|person|people|boy|girl|lady|gentleman|guy|individual|human|model|face|portrait|figure|child|baby)\b",
    re.IGNORECASE)


def is_person_prompt(text):
    return _PERSON_TOKEN_RE.search(text) is not None


# ======================================================================================
# Main entry point
# ======================================================================================
def augment(source_prompt: str, row_id, cfg: PromptAugmentConfig = None) -> dict:
    cfg = cfg or PromptAugmentConfig()
    text = (source_prompt or "").strip()
    seed = stable_seed(row_id, cfg.seed_salt)
    rng = random.Random(seed)
    meta = {
        "id": str(row_id), "seed": seed, "source_prompt": source_prompt, "final_prompt": text,
        "race": None, "race_injected": False, "is_tail": False, "gender": None,
        "hair": None, "hair_injected": False, "eye": None, "eye_injected": False,
        "expression": None, "makeup": None, "jewelry": None, "is_amateur": False,
        "age_band": None, "age_overridden": False, "age_terms_removed": [],
        "is_face_prompt": False, "policy_version": POLICY_VERSION,
    }
    if not text:
        return meta
    is_person = is_person_prompt(text)
    meta["is_face_prompt"] = is_person
    if cfg.inject_only_if_person and not is_person:
        return meta

    race_label, _ = detect_race(text, cfg)
    gender, _minor = detect_gender(text)
    meta["gender"] = gender
    hair_src = detect_hair(text)
    eye_src = has_eye(text)
    makeup_src = has_makeup(text)
    jewelry_src = has_jewelry(text)

    race_explicit = race_label is not None
    label, is_tail = (race_label, False) if race_explicit else sample_race(rng, cfg)
    meta["race"] = label
    meta["race_injected"] = not race_explicit
    meta["is_tail"] = is_tail

    # base text: strip age + expression + B&W, de-minor
    base, removed = strip_age_terms(text)
    base = strip_expression(base)
    base = strip_bw(base)
    base = deminor(base)
    meta["age_band"] = cfg.age_band
    meta["age_overridden"] = len(removed) > 0
    meta["age_terms_removed"] = removed

    # inject race into the subject in place (only when not already stated)
    if not race_explicit:
        base, _ok = _inject_race_into_subject(base, _race_descriptor(label, gender))

    # assemble appended clauses (only inject what the source did not specify)
    clauses = [_age_phrase(gender)]

    if cfg.inject_hair and hair_src is None:
        pal = HAIR_DARK if label in DARK_FEATURE_RACES else HAIR_FULL
        hair = sample_weighted(rng, pal)
        clauses.append(f"with {hair}")
        meta["hair"], meta["hair_injected"] = hair, True
    else:
        meta["hair"] = hair_src

    if cfg.inject_eye and not eye_src:
        pal = EYE_DARK if label in DARK_FEATURE_RACES else EYE_FULL
        eye = sample_weighted(rng, pal)
        clauses.append(f"with {eye}")
        meta["eye"], meta["eye_injected"] = eye, True

    if cfg.inject_expression:
        expr = sample_weighted(rng, EXPRESSIONS)     # ALWAYS (overrides stripped source smile)
        clauses.append(expr)
        meta["expression"] = expr

    if cfg.inject_makeup and gender == "woman" and not makeup_src:
        mk = sample_weighted(rng, MAKEUP)
        clauses.append(mk)
        meta["makeup"] = mk

    if cfg.inject_jewelry and not jewelry_src:
        jw = sample_weighted(rng, JEWELRY_WOMEN if gender == "woman" else JEWELRY_MEN)
        clauses.append(jw)
        meta["jewelry"] = jw

    # quality / color tier (always); ~1/10 amateur
    is_amateur = rng.random() < cfg.amateur_fraction
    quality = sample_weighted(rng, QUALITY_AMATEUR if is_amateur else QUALITY_PHOTO)
    meta["is_amateur"] = is_amateur

    final = base.rstrip(" .,") + ", " + ", ".join(clauses) + ". " + quality + "."
    meta["final_prompt"] = _sentence_case(_tidy(final)).rstrip(".") + "."
    return meta


def _sentence_case(text):
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


# ======================================================================================
# Anti-flooding: token-shingle MinHash/LSH "sentence similarity" guard
# ======================================================================================
import zlib  # noqa: E402  (stable, non-salted hash for shingles)

# Boilerplate / structural tokens shared by ~every prompt -> excluded from the similarity
# signature so it measures the VARIABLE content (source body + injected attributes), which is
# where "too many matching tokens" actually shows up.
_GUARD_STOP = set(
    "a an the of in on at with and to is are was were be been has have had not her his their she "
    "he they them image features shows depicts displays person people man woman men women male "
    "female individual face portrait photograph photo color colour realistic photorealistic "
    "professional studio dslr headshot quality high natural lighting light soft balanced detail "
    "fine sharp focus depth field skin texture tones true life expression look gaze wearing with "
    "late twenties early thirties aged adult years old amateur snapshot candid casual phone "
    "everyday slightly".split()
)


def _crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


class DiversityGuard:
    """Detect near-duplicate prompts (high token-shingle overlap) so the caller can resample
    attributes and avoid flooding the dataset with too-similar text. Dependency-free MinHash + LSH;
    stateful per process (per rank slice). Approximate but cheap (O(1) amortized per prompt)."""

    def __init__(self, num_perm=32, bands=8, threshold=0.82):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = max(1, num_perm // bands)
        self.threshold = threshold
        self._a = [(2654435761 * (i + 1) + 1) & 0xFFFFFFFF for i in range(num_perm)]
        self._b = [(40503 * (i + 1) + 7) & 0xFFFFFFFF for i in range(num_perm)]
        self.buckets = [dict() for _ in range(bands)]

    def _shingles(self, text):
        toks = [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 2 and t not in _GUARD_STOP]
        sh = set(toks)
        sh |= {toks[i] + "_" + toks[i + 1] for i in range(len(toks) - 1)}
        return sh

    def _sig(self, shingles):
        if not shingles:
            return [0] * self.num_perm
        hs = [_crc(s) for s in shingles]
        return [min(((a * h + b) & 0xFFFFFFFF) for h in hs) for a, b in zip(self._a, self._b)]

    def is_duplicate(self, text) -> bool:
        sh = self._shingles(text)
        if not sh:
            return False
        sig = self._sig(sh)
        for bi in range(self.bands):
            band = tuple(sig[bi * self.rows:(bi + 1) * self.rows])
            rep = self.buckets[bi].get(band)
            if rep is not None:
                inter = len(sh & rep)
                union = len(sh | rep) or 1
                if inter / union >= self.threshold:
                    return True
        return False

    def add(self, text):
        sh = self._shingles(text)
        if not sh:
            return
        sig = self._sig(sh)
        for bi in range(self.bands):
            band = tuple(sig[bi * self.rows:(bi + 1) * self.rows])
            self.buckets[bi].setdefault(band, sh)


def augment_diverse(source_prompt, row_id, cfg=None, guard=None, max_retries=4):
    """augment() + optional anti-flooding: if the result is a near-duplicate of an earlier prompt
    (per `guard`), resample attributes by perturbing the seed, up to max_retries. Records how many
    resamples were needed in meta['dedup_resamples']."""
    m = augment(source_prompt, row_id, cfg)
    if guard is None or not m["final_prompt"]:
        m["dedup_resamples"] = 0
        return m
    tries = 0
    while tries < max_retries and guard.is_duplicate(m["final_prompt"]):
        tries += 1
        m = augment(source_prompt, f"{row_id}#dz{tries}", cfg)
    guard.add(m["final_prompt"])
    m["dedup_resamples"] = tries
    return m


if __name__ == "__main__":
    from collections import Counter
    cfg = PromptAugmentConfig()
    examples = [
        ("The image features a young man with a round face and a wide smile, wearing a yellow hat and raincoat.", "ex1"),
        ("The woman in the image is of Asian descent, likely in her late 20s to early 40s, with big blue eyes and a small nose, smiling.", "ex2"),
        ("The image features a young Caucasian girl with a heart-shaped face, blonde hair, and a bright smile, wearing earrings.", "ex3"),
        ("A man wearing a gray suit, standing in an office.", "ex4"),
        ("An elderly man with a long white beard, grinning broadly.", "ex5"),
    ]
    for src, rid in examples:
        m = augment(src, rid, cfg)
        print(f"\nid={rid}\n  SRC: {src}\n  OUT: {m['final_prompt']}")
        print(f"  race={m['race']}({'inj' if m['race_injected'] else 'src'}) gender={m['gender']} "
              f"hair={m['hair']} eye={m['eye']} expr={m['expression']!r} makeup={m['makeup']!r} "
              f"jewelry={m['jewelry']!r} amateur={m['is_amateur']}")
    N = 6000
    ex = Counter(augment("a person", f"e{i}", cfg)["expression"] for i in range(N))
    print(f"\nexpression spread over {N} (top + smile share):")
    sm = sum(v for k, v in ex.items() if k and ("smil" in k or "laugh" in k))
    print(f"  smile-family: {100*sm/N:.1f}%  | distinct expressions: {len(ex)}")
    for k, v in ex.most_common(6):
        print(f"    {k:34s} {100*v/N:4.1f}%")
    am = sum(augment("a person", f"q{i}", cfg)["is_amateur"] for i in range(N))
    print(f"  amateur fraction: {100*am/N:.1f}%")
