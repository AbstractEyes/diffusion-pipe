"""Prompt augmentation / bias-mitigation policy for Qwen-Image synthetic-character generation.

Qwen-Image, given a prompt with no explicit race/ethnicity, collapses toward asian-looking
faces. Left unchecked, a dataset synthesized from generic captions becomes a single race. This
module rewrites each source prompt so the synthetic set spans a fair, configurable race mix and
is age-constrained to adults (~25-35), while RESPECTING any race/age the source prompt already
states.

Design notes:
- Race is SAMPLED (per a tunable, more-uniform distribution) only when the source prompt does
  not already specify one. Explicit race is kept verbatim.
- Age terms (minor / elderly / explicit numbers) are always stripped and a 25-35 adult phrase is
  injected. A separate strong age-verification filter runs LATER (not here); this only constrains
  the prompt and records metadata.
- "Default" man/woman -> brunette Caucasian: Caucasian is the single highest-weighted race, and
  brunette hair is injected only when hair is unspecified AND the sampled race is the Caucasian
  default.
- Everything is deterministic per row id (blake2b seed, not Python's salted hash()), so race
  assignment and (downstream) image seed are reproducible across machines/runs.

Dependency-light: stdlib only (re, random, hashlib, dataclasses). Unit-testable without torch.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

POLICY_VERSION = "augment-v1"


# --------------------------------------------------------------------------------------
# Taxonomy + descriptors
# --------------------------------------------------------------------------------------
# Natural-language descriptor injected before the gender noun, e.g. "a Caucasian woman".
# hispanic_latino is gendered at render time (Latina/Latino/Hispanic).
RACE_DESCRIPTORS = {
    "caucasian": "Caucasian",
    "east_asian": "East Asian",
    "south_asian": "South Asian",
    "southeast_asian": "Southeast Asian",
    "black": "Black",
    "hispanic_latino": "Hispanic",
    "middle_eastern": "Middle Eastern",
    "native_american": "Native American",
    "pacific_islander": "Pacific Islander",
    "multiracial": "mixed-race",
    # rare tail (kept deliberately small for later small-bucket re-imposition)
    "central_asian": "Central Asian",
    "mediterranean": "Mediterranean",
    "persian": "Persian",
    "ethiopian": "Ethiopian",
    "scandinavian": "Scandinavian",
    "polynesian": "Polynesian",
    "mestizo": "Mestizo",
}

# More-uniform distribution (sums to 100): Caucasian slightly highest (preserves "default =
# Caucasian"), broad spread across the top-10, and a larger ~15% rare tail.
TOP10_WEIGHTS = {
    "caucasian": 15,
    "east_asian": 8,
    "black": 8,
    "south_asian": 8,
    "hispanic_latino": 8,
    "middle_eastern": 8,
    "southeast_asian": 8,
    "native_american": 8,
    "pacific_islander": 7,
    "multiracial": 7,
}
TAIL_WEIGHTS = {
    "central_asian": 3,
    "mediterranean": 2,
    "persian": 2,
    "ethiopian": 2,
    "scandinavian": 2,
    "polynesian": 2,
    "mestizo": 2,
}

# Detection phrases -> normalized label. Longest-match-first is applied at search time. Bare
# "white"/"black" are guarded (color vs race) separately.
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
    # tail surfaces
    "central_asian": ["central asian", "kazakh", "uzbek", "afghan"],
    "mediterranean": ["mediterranean", "greek", "italian", "spanish", "portuguese"],
    "persian": ["persian", "iranian"],
    "ethiopian": ["ethiopian", "eritrean", "somali"],
    "scandinavian": ["scandinavian", "swedish", "norwegian", "danish", "finnish", "icelandic"],
    "polynesian": ["polynesian", "tongan", "fijian"],
    "mestizo": ["mestizo", "mestiza"],
}

PERSON_NOUNS = ["man", "men", "woman", "women", "male", "female", "person", "people", "boy",
                "girl", "lady", "gentleman", "guy", "individual", "human", "figure", "model",
                "skin", "complexion", "descent"]

WOMAN_TOKENS = ["woman", "women", "female", "lady", "girl", "girls", "mother", "wife", "she"]
MAN_TOKENS = ["man", "men", "male", "gentleman", "guy", "boy", "boys", "father", "husband"]
MINOR_GENDER_TOKENS = {"girl", "girls", "boy", "boys"}

HAIR_COLORS = ["blonde", "blond", "brunette", "auburn", "redhead", "red-haired", "ginger",
               "brown hair", "black hair", "dark hair", "light hair", "red hair", "gray hair",
               "grey hair", "white hair", "silver hair", "platinum"]

AGE_MINOR_TERMS = ["youthful", "young", "teenage", "teenager", "teen", "adolescent", "children",
                   "child", "kids", "kid", "baby", "infant", "toddler", "juvenile", "minor",
                   "schoolgirl", "schoolboy", "little girl", "little boy", "underage", "preteen"]
AGE_OLD_TERMS = ["elderly", "middle-aged", "middle aged", "geriatric", "senior", "wrinkled",
                 "gray-haired", "grey-haired", "white-haired", "grandmother", "grandfather",
                 "grandma", "grandpa", "aged", "old"]
# numeric / decade age phrases
AGE_NUMERIC_PATTERNS = [
    r"\b\d{1,2}\s*[-\s]?\s*years?[\s-]?old\b",
    r"\baged?\s+\d{1,2}\b",
    r"\bin\s+(his|her|their)\s+(teens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
]


@dataclass
class PromptAugmentConfig:
    top10_weights: dict = field(default_factory=lambda: dict(TOP10_WEIGHTS))
    tail_weights: dict = field(default_factory=lambda: dict(TAIL_WEIGHTS))
    default_race: str = "caucasian"
    default_hair: str = "brunette"
    brunette_only_for_default_caucasian: bool = True
    age_phrase_person: str = "in their late twenties to early thirties"
    age_phrase_woman: str = "in her late twenties to early thirties"
    age_phrase_man: str = "in his late twenties to early thirties"
    age_band: str = "25-35"
    inject_only_if_person: bool = True   # leave non-person prompts unchanged
    asian_generic_to: str = "east_asian"  # bare "asian" -> this label
    seed_salt: str = "qwen-synth-v1"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def stable_seed(row_id, salt: str = "qwen-synth-v1") -> int:
    """Deterministic seed from a row id string (NOT Python's salted hash()). Masked to 63 bits so
    it always fits a signed int64 (parquet column / torch manual_seed)."""
    h = hashlib.blake2b(f"{salt}:{row_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def _word_re(phrase: str) -> re.Pattern:
    # word-boundaried, spaces -> flexible whitespace/hyphen
    esc = re.escape(phrase).replace(r"\ ", r"[\s-]+")
    return re.compile(rf"\b{esc}\b", re.IGNORECASE)


# precompute sorted (phrase, label) longest-first, and per-phrase regex
_PHRASE_LABELS = []
for _label, _phrases in RACE_PHRASES.items():
    for _p in _phrases:
        _PHRASE_LABELS.append((_p, _label))
_PHRASE_LABELS.sort(key=lambda pl: -len(pl[0]))
_PHRASE_RE = {p: _word_re(p) for p, _ in _PHRASE_LABELS}
_PERSON_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in PERSON_NOUNS) + r")\b", re.IGNORECASE)


# "white"/"black" count as RACE only in these tight forms (else they are colors: white hat,
# black hair, black dress). Precise adjacency, not a loose window.
_COLOR_RACE_RE = {
    "white": re.compile(
        r"\b(white\s+(?:man|men|woman|women|male|female|person|people|lady|gentleman|guy|individual|skin|complexion)"
        r"|(?:man|woman|male|female|person|skin|complexion)\s+(?:who\s+is\s+|is\s+)?white)\b", re.IGNORECASE),
    "black": re.compile(
        r"\b(black\s+(?:man|men|woman|women|male|female|person|people|lady|gentleman|guy|individual|skin|complexion)"
        r"|(?:man|woman|male|female|person|skin|complexion)\s+(?:who\s+is\s+|is\s+)?black)\b", re.IGNORECASE),
}


def detect_race(text: str, cfg: PromptAugmentConfig):
    """Return (label, surface) for an explicitly-stated race, else (None, None)."""
    # "of X descent"
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


def _normalize_label(label: str, cfg: PromptAugmentConfig) -> str:
    if label == "east_asian":
        return cfg.asian_generic_to  # configurable mapping of generic "asian"
    return label


def detect_gender(text: str):
    """Return (gender in {'woman','man','person'}, is_minor_token)."""
    best = None  # (pos, gender, minor)
    for token, gender in [(t, "woman") for t in WOMAN_TOKENS] + [(t, "man") for t in MAN_TOKENS]:
        m = _word_re(token).search(text)
        if m:
            cand = (m.start(), gender, token in MINOR_GENDER_TOKENS)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return "person", False
    return best[1], best[2]


def detect_hair(text: str):
    for color in sorted(HAIR_COLORS, key=lambda c: -len(c)):
        if _word_re(color).search(text):
            return color
    return None


def strip_age_terms(text: str):
    """Remove minor/elderly/numeric age cues. Returns (cleaned, removed_terms)."""
    removed = []
    out = text
    for pat in AGE_NUMERIC_PATTERNS:
        for m in re.finditer(pat, out, re.IGNORECASE):
            removed.append(m.group(0))
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    for term in sorted(AGE_MINOR_TERMS + AGE_OLD_TERMS, key=lambda t: -len(t)):
        rx = _word_re(term)
        if rx.search(out):
            removed.append(term)
            out = rx.sub("", out)
    out = _tidy(out)
    return out, removed


_DEMINOR = [
    (re.compile(r"\blittle\s+girls?\b", re.I), "woman"),
    (re.compile(r"\blittle\s+boys?\b", re.I), "man"),
    (re.compile(r"\bschoolgirls?\b", re.I), "woman"),
    (re.compile(r"\bschoolboys?\b", re.I), "man"),
    (re.compile(r"\bgirls\b", re.I), "women"),
    (re.compile(r"\bboys\b", re.I), "men"),
    (re.compile(r"\bgirl\b", re.I), "woman"),
    (re.compile(r"\bboy\b", re.I), "man"),
]


def deminor(text: str) -> str:
    """Replace minor gender nouns with adult equivalents (age-safety; nothing younger survives)."""
    for rx, repl in _DEMINOR:
        text = rx.sub(repl, text)
    return text


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,;:])", r"\1", text)
    text = re.sub(r"\ban\s+(?=[^aeiouAEIOU\s])", "a ", text, flags=re.IGNORECASE)  # "An man" -> "a man"
    text = re.sub(r"\ba\s+(?=[aeiouAEIOU])", "an ", text, flags=re.IGNORECASE)     # "a East Asian" -> "an East Asian"
    text = re.sub(r"\b(a|an|the)\s+(,|\.|with|wearing|and)\b", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    return text.strip(" ,;")


_PREAMBLE_RE = [
    re.compile(r"^\s*the\s+image\s+(features|shows|depicts|displays|contains|is\s+of)\s+", re.IGNORECASE),
    re.compile(r"^\s*(this\s+is\s+)?(a|an)\s+(photo|picture|portrait|image|close[- ]up|headshot|photograph)\s+of\s+", re.IGNORECASE),
    re.compile(r"^\s*(the|a|an)\s+(man|woman|person|boy|girl|individual|figure|model|lady|gentleman)\s+(in\s+the\s+image\s+)?(is|has|wears|is\s+wearing|wearing)\s+", re.IGNORECASE),
]


def _strip_preamble(text: str) -> str:
    for rx in _PREAMBLE_RE:
        m = rx.match(text)
        if m:
            return text[m.end():].strip()
    # also drop a bare leading article + gender noun "a woman with ..." -> "with ..."
    m = re.match(r"^\s*(a|an|the)\s+([\w\- ]*?)?(man|woman|person|boy|girl|lady|gentleman|individual)\b\s*", text, re.IGNORECASE)
    if m and m.end() < len(text):
        return text[m.end():].strip()
    return text.strip()


def _article(first_word: str) -> str:
    return "an" if first_word[:1].lower() in "aeiou" else "a"


def _race_descriptor(label: str, gender: str) -> str:
    if label == "hispanic_latino":
        return {"woman": "Latina", "man": "Latino"}.get(gender, "Hispanic")
    return RACE_DESCRIPTORS[label]


def _gender_noun(gender: str) -> str:
    return {"woman": "woman", "man": "man"}.get(gender, "person")


def _age_phrase(gender: str, cfg: PromptAugmentConfig) -> str:
    return {"woman": cfg.age_phrase_woman, "man": cfg.age_phrase_man}.get(gender, cfg.age_phrase_person)


def sample_race(rng: random.Random, cfg: PromptAugmentConfig):
    """Weighted draw over top-10 + tail. Returns (label, is_tail)."""
    combined = list(cfg.top10_weights.items()) + list(cfg.tail_weights.items())
    combined.sort(key=lambda kv: kv[0])  # fixed order for determinism
    total = sum(w for _, w in combined)
    u = rng.random() * total
    acc = 0.0
    for label, w in combined:
        acc += w
        if u < acc:
            return label, (label in cfg.tail_weights)
    return combined[-1][0]  # fallback


_PERSON_TOKEN_RE = re.compile(
    r"\b(man|men|woman|women|male|female|person|people|boy|girl|lady|gentleman|guy|individual|human|model|face|portrait|figure|child|baby)\b",
    re.IGNORECASE,
)


def is_person_prompt(text: str) -> bool:
    return _PERSON_TOKEN_RE.search(text) is not None


_SUBJECT_RE = re.compile(
    r"\b(a|an|the)\s+((?:[a-z]+\s+){0,3}?)(men|man|women|woman|male|female|people|person|lady|gentleman|guy|individual)\b",
    re.IGNORECASE,
)
_ADULT_NOUN = {"male": "man", "female": "woman", "lady": "woman", "gentleman": "man",
               "guy": "man", "individual": "person"}


def _inject_race_into_subject(text: str, descriptor: str, hair_adj: str):
    """Insert '<hair_adj><descriptor> ' before the first subject noun, in place, preserving any
    existing adjectives (e.g. 'a blonde woman' -> 'a blonde Southeast Asian woman'). Returns
    (new_text, injected_bool)."""
    m = _SUBJECT_RE.search(text)
    if not m:
        return text, False
    article, adjs, noun = m.group(1), m.group(2) or "", m.group(3)
    adult = _ADULT_NOUN.get(noun.lower(), noun)
    new_subject = f"{article} {adjs}{hair_adj}{descriptor} {adult}"
    return text[:m.start()] + new_subject + text[m.end():], True


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def augment(source_prompt: str, row_id, cfg: PromptAugmentConfig = None) -> dict:
    """Rewrite a source prompt for fair race coverage + adult 25-35 age. Deterministic per row_id.

    Returns a metadata dict including 'final_prompt' (the prompt to generate with) and the
    decisions made, suitable as parquet columns.
    """
    cfg = cfg or PromptAugmentConfig()
    text = (source_prompt or "").strip()
    seed = stable_seed(row_id, cfg.seed_salt)
    rng = random.Random(seed)

    meta = {
        "id": str(row_id),
        "seed": seed,
        "source_prompt": source_prompt,
        "final_prompt": text,
        "race": None,
        "race_injected": False,
        "is_tail": False,
        "gender": None,
        "hair": None,
        "hair_injected": False,
        "age_band": None,
        "age_overridden": False,
        "age_terms_removed": [],
        "is_face_prompt": False,
        "policy_version": POLICY_VERSION,
    }

    if not text:
        return meta

    is_person = is_person_prompt(text)
    meta["is_face_prompt"] = is_person
    if cfg.inject_only_if_person and not is_person:
        return meta  # leave scenes/objects untouched

    race_label, race_surface = detect_race(text, cfg)
    gender, _minor = detect_gender(text)
    hair = detect_hair(text)
    meta["gender"] = gender

    race_explicit = race_label is not None
    if race_explicit:
        label, is_tail = race_label, False
    else:
        label, is_tail = sample_race(rng, cfg)
    meta["race"] = label
    meta["race_injected"] = not race_explicit
    meta["is_tail"] = is_tail

    # age: always strip conflicting cues; de-minor; inject adult band
    cleaned, removed = strip_age_terms(text)
    cleaned = deminor(cleaned)
    age_phrase = _age_phrase(gender, cfg)
    meta["age_band"] = cfg.age_band
    meta["age_overridden"] = len(removed) > 0
    meta["age_terms_removed"] = removed

    # brunette default only for sampled Caucasian default with no stated hair
    inject_hair = bool(
        hair is None
        and label == cfg.default_race
        and (not race_explicit or not cfg.brunette_only_for_default_caucasian)
        and cfg.default_hair
    )
    meta["hair"] = hair if hair else (cfg.default_hair if inject_hair else None)
    meta["hair_injected"] = inject_hair

    if race_explicit:
        # keep source race; just ensure the adult age band is present
        final = cleaned.rstrip(" .") + f", {age_phrase}."
        meta["final_prompt"] = _sentence_case(_tidy(final))
        return meta

    # injected race: insert "<hair?> <race> " before the existing subject noun, in place
    descriptor = _race_descriptor(label, gender)
    hair_adj = (cfg.default_hair + " ") if inject_hair else ""
    injected_text, ok = _inject_race_into_subject(cleaned, descriptor, hair_adj)
    if ok:
        final = injected_text.rstrip(" .") + f", {age_phrase}."
    else:
        # no explicit subject noun (e.g. "close-up portrait, ..."): front-load a fresh subject
        noun = _gender_noun(gender)
        first_word = (hair_adj or descriptor).split()[0]
        lead = f"{_article(first_word)} {hair_adj}{descriptor} {noun} {age_phrase}"
        final = f"{lead}. {cleaned}" if cleaned else lead
    meta["final_prompt"] = _sentence_case(_tidy(final)).rstrip(".") + "."
    return meta


def _sentence_case(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


if __name__ == "__main__":
    cfg = PromptAugmentConfig()
    examples = [
        ("A man wearing a gray suit, standing in an office.", "ex1"),
        ("The woman in the image is of Asian descent, has a round face, and is wearing glasses.", "ex2"),
        ("The image features a young blonde woman with a pink hat, wearing a red and white outfit.", "ex3"),
        ("The image features a blonde woman with blue eyes, wearing a green dress.", "ex4"),
        ("A close-up of a red sports car on a mountain road at sunset.", "ex5"),
        ("An elderly man with a long white beard and wrinkled face.", "ex6"),
    ]
    for src, rid in examples:
        m = augment(src, rid, cfg)
        print(f"\nid={rid}")
        print("  SRC :", src)
        print("  OUT :", m["final_prompt"])
        print(f"  meta: race={m['race']} injected={m['race_injected']} tail={m['is_tail']} "
              f"gender={m['gender']} hair={m['hair']}({'inj' if m['hair_injected'] else 'src'}) "
              f"age_removed={m['age_terms_removed']}")
    # rough distribution check
    from collections import Counter
    c = Counter(augment("A person standing.", f"row{i}", cfg)["race"] for i in range(4000))
    print("\nsampled race distribution over 4000 generic prompts:")
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {100*v/4000:5.1f}%")
