"""Unit tests for qwen_extraction/prompt_policy.py (v2: race + expression/attribute subtext system).

Runs in the CPU testvenv (stdlib only): python test/test_prompt_policy.py
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qwen_extraction import prompt_policy as pp

cfg = pp.PromptAugmentConfig()
SMILE_RE = re.compile(r"\b(smil\w*|grin\w*|laugh\w*)\b", re.I)
CHILD_RE = re.compile(r"\b(girl|boy|child|children|kid|kids|young|youthful|teen\w*|baby|toddler|infant|minor)\b", re.I)
BW_RE = re.compile(r"\b(monochrome|greyscale|grayscale|black and white|sepia)\b", re.I)


def _w(text, w):
    return re.search(rf"\b{re.escape(w)}\b", text, re.I) is not None


def test_race_detection():
    assert pp.detect_race("a white man standing", cfg)[0] == 'caucasian'
    assert pp.detect_race("a black woman smiling", cfg)[0] == 'black'
    assert pp.detect_race("the woman is of Asian descent", cfg)[0] == cfg.asian_generic_to
    assert pp.detect_race("an Indian man", cfg)[0] == 'south_asian'
    assert pp.detect_race("wearing a white hat", cfg)[0] is None        # color, not race
    assert pp.detect_race("with long black hair and a black dress", cfg)[0] is None
    assert pp.detect_race("a man with a white beard", cfg)[0] is None
    print("test_race_detection OK")


def test_expression_override():
    # a source that says "smiling" must NOT keep the smile; an expression is injected
    m = pp.augment("The image features a woman with a wide smile, wearing a hat.", "e1", cfg)
    assert m["expression"] is not None
    assert not SMILE_RE.search(m["final_prompt"]) or "smil" in (m["expression"] or "").lower(), m["final_prompt"]
    # over many generic prompts, smile-family is a minority (was ~67% in the source set)
    exprs = [pp.augment("a person", f"x{i}", cfg)["expression"] for i in range(4000)]
    sm = sum(1 for e in exprs if e and SMILE_RE.search(e))
    assert sm / 4000 < 0.30, sm / 4000
    assert len(set(exprs)) >= 20, len(set(exprs))           # broad spread
    print(f"test_expression_override OK (smile-family={100*sm/4000:.1f}%, {len(set(exprs))} expressions)")


def test_attribute_injection_and_respect():
    # generic source lacking attributes -> hair/eye/expression injected; woman gets makeup; jewelry set
    m = pp.augment("a woman standing in a room", "a1", cfg)
    assert m["hair"] and m["eye"] and m["expression"] and m["makeup"] and m["jewelry"]
    assert m["hair_injected"] and m["eye_injected"]
    # explicit attributes are respected (not re-injected)
    m2 = pp.augment("a woman with blonde hair, blue eyes, wearing earrings and red lipstick", "a2", cfg)
    assert m2["hair"] == "blonde" and m2["hair_injected"] is False
    assert m2["eye"] is None            # source had eyes -> not injected
    assert m2["makeup"] is None and m2["jewelry"] is None   # source had lipstick + earrings
    print("test_attribute_injection_and_respect OK")


def test_quality_color_no_bw():
    for i in range(200):
        f = pp.augment("a man in a city", f"q{i}", cfg)["final_prompt"]
        assert "color" in f.lower(), f            # color enforced
        assert not BW_RE.search(f), f             # never monochrome/greyscale
    # B&W in the source is stripped
    m = pp.augment("a black and white photo of a woman", "bw", cfg)
    assert not BW_RE.search(m["final_prompt"]), m["final_prompt"]
    print("test_quality_color_no_bw OK")


def test_amateur_fraction():
    am = sum(pp.augment("a person", f"am{i}", cfg)["is_amateur"] for i in range(4000))
    assert 0.06 < am / 4000 < 0.15, am / 4000
    print(f"test_amateur_fraction OK ({100*am/4000:.1f}%)")


def test_child_safety_not_encouraged():
    # de-minor + age strip: nothing child-related survives (we don't filter, we just don't encourage)
    for src in ["a young girl with pigtails", "a little boy playing", "a teenage girl smiling"]:
        m = pp.augment(src, "c_" + src[:4], cfg)
        assert not CHILD_RE.search(m["final_prompt"]), (src, m["final_prompt"])
        assert m["age_band"] == "25-35"
    print("test_child_safety_not_encouraged OK")


def test_race_distribution():
    N = 5000
    c = Counter(pp.augment("a person.", f"r{i}", cfg)["race"] for i in range(N))
    frac = {k: v / N for k, v in c.items()}
    assert max(frac, key=frac.get) == 'caucasian', frac
    assert 0.11 < frac['caucasian'] < 0.19, frac['caucasian']
    tail = sum(frac.get(k, 0) for k in pp.TAIL_WEIGHTS)
    assert 0.10 < tail < 0.20, tail
    print(f"test_race_distribution OK (caucasian={frac['caucasian']:.2f}, tail={tail:.2f})")


def test_determinism():
    assert pp.augment("a man in a suit", "id9", cfg) == pp.augment("a man in a suit", "id9", cfg)
    assert pp.stable_seed("x") == pp.stable_seed("x")
    print("test_determinism OK")


def test_diversity_guard():
    g = pp.DiversityGuard(threshold=0.8)
    a = "a tall Caucasian woman with long blonde hair and green eyes wearing a red dress in a garden"
    c = "a Black man with a thick beard playing electric guitar on a busy city street at night"
    assert not g.is_duplicate(a)
    g.add(a)
    assert g.is_duplicate(a)            # identical -> duplicate
    assert not g.is_duplicate(c)        # very different -> not duplicate
    m = pp.augment_diverse("a woman walking in a park", "dz", cfg, guard=pp.DiversityGuard())
    assert "dedup_resamples" in m and m["final_prompt"]
    print("test_diversity_guard OK")


ALL = [test_race_detection, test_expression_override, test_attribute_injection_and_respect,
       test_quality_color_no_bw, test_amateur_fraction, test_child_safety_not_encouraged,
       test_race_distribution, test_determinism, test_diversity_guard]

if __name__ == '__main__':
    for t in ALL:
        t()
    print(f"\nAll {len(ALL)} prompt_policy tests passed.")
