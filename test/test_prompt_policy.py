"""Unit tests for qwen_extraction/prompt_policy.py (race/age bias-mitigation).

Runs in the CPU testvenv (stdlib only): python test/test_prompt_policy.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qwen_extraction import prompt_policy as pp

cfg = pp.PromptAugmentConfig()
BANNED_AGE = ['young', 'teen', 'teenager', 'child', 'kid', 'elderly', 'old', 'wrinkled',
              'girl', 'boy', 'schoolgirl', 'baby', 'toddler']


def _has_word(text, w):
    import re
    return re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) is not None


def test_race_detection():
    assert pp.detect_race("a white man standing", cfg)[0] == 'caucasian'
    assert pp.detect_race("a black woman smiling", cfg)[0] == 'black'
    assert pp.detect_race("the woman is of Asian descent", cfg)[0] == cfg.asian_generic_to
    assert pp.detect_race("an Indian man", cfg)[0] == 'south_asian'
    assert pp.detect_race("a Latina woman", cfg)[0] == 'hispanic_latino'
    # color-vs-race guard: these are NOT races
    assert pp.detect_race("wearing a white hat", cfg)[0] is None
    assert pp.detect_race("with long black hair and a black dress", cfg)[0] is None
    assert pp.detect_race("a man with a white beard", cfg)[0] is None
    print("test_race_detection OK")


def test_deminor_and_age():
    cleaned, removed = pp.strip_age_terms("a young woman, 22 years old, youthful face")
    assert 'young' in removed and not _has_word(cleaned, 'young')
    assert not any(c.isdigit() for c in cleaned), cleaned
    assert pp.deminor("a little girl and a boy") == "a woman and a man"
    print("test_deminor_and_age OK")


def test_explicit_race_respected_age_added():
    m = pp.augment("The woman in the image is of Asian descent, wearing glasses.", "r1", cfg)
    assert m['race'] == cfg.asian_generic_to and m['race_injected'] is False
    assert 'descent' in m['final_prompt'].lower()           # source race kept
    assert 'twenties' in m['final_prompt'].lower()          # age injected
    print("test_explicit_race_respected_age_added OK")


def test_injected_race_inplace_no_dup_no_minor():
    m = pp.augment("The image features a young blonde woman with a pink hat.", "r2", cfg)
    out = m['final_prompt']
    assert m['race_injected'] is True
    assert m['race'].replace('_', ' ') in out.lower() or pp.RACE_DESCRIPTORS[m['race']].lower() in out.lower()
    assert out.lower().count('woman') == 1, out               # no duplicated subject
    assert 'blonde' in out.lower()                            # existing hair preserved
    for w in BANNED_AGE:
        assert not _has_word(out, w), (w, out)
    print("test_injected_race_inplace_no_dup_no_minor OK")


def test_non_person_unchanged():
    src = "A close-up of a red sports car on a mountain road at sunset."
    m = pp.augment(src, "r3", cfg)
    assert m['final_prompt'] == src and m['race'] is None
    print("test_non_person_unchanged OK")


def test_brunette_only_for_default_caucasian():
    # find a generic prompt id whose sampled race is caucasian; brunette must be injected
    for i in range(200):
        m = pp.augment("a person standing in a room", f"cauc{i}", cfg)
        if m['race'] == 'caucasian':
            assert m['hair'] == 'brunette' and m['hair_injected'] is True
            assert 'brunette' in m['final_prompt'].lower()
            break
    else:
        raise AssertionError("no caucasian sample found in 200 draws")
    # a non-caucasian sample must NOT get brunette injected
    for i in range(200):
        m = pp.augment("a person standing in a room", f"other{i}", cfg)
        if m['race'] != 'caucasian':
            assert m['hair_injected'] is False
            break
    print("test_brunette_only_for_default_caucasian OK")


def test_determinism():
    a = pp.augment("a man in a suit", "same-id", cfg)
    b = pp.augment("a man in a suit", "same-id", cfg)
    assert a == b
    assert pp.stable_seed("x") == pp.stable_seed("x")
    print("test_determinism OK")


def test_distribution():
    N = 6000
    c = Counter(pp.augment("a person.", f"row{i}", cfg)['race'] for i in range(N))
    frac = {k: v / N for k, v in c.items()}
    # caucasian is the single highest, ~15%
    assert max(frac, key=frac.get) == 'caucasian', frac
    assert 0.11 < frac['caucasian'] < 0.19, frac['caucasian']
    # tail labels present and collectively ~15%
    tail = sum(frac.get(k, 0) for k in pp.TAIL_WEIGHTS)
    assert 0.10 < tail < 0.20, tail
    # no single non-caucasian top-10 race saturates
    for k in pp.TOP10_WEIGHTS:
        if k != 'caucasian':
            assert frac.get(k, 0) < 0.12, (k, frac.get(k, 0))
    print(f"test_distribution OK (caucasian={frac['caucasian']:.2f}, tail={tail:.2f})")


ALL = [test_race_detection, test_deminor_and_age, test_explicit_race_respected_age_added,
       test_injected_race_inplace_no_dup_no_minor, test_non_person_unchanged,
       test_brunette_only_for_default_caucasian, test_determinism, test_distribution]

if __name__ == '__main__':
    for t in ALL:
        t()
    print(f"\nAll {len(ALL)} prompt_policy tests passed.")
