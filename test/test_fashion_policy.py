"""Unit tests for the fashion domain (DeepFashion expansion): fashion_captions + augment_fashion.

Runs in the CPU testvenv: python test/test_fashion_policy.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qwen_extraction import fashion_captions as fc, prompt_policy as pp

cfg = pp.PromptAugmentConfig(domain="fashion")
FULL_RE = re.compile(r"full[- ]length|full[- ]body|head[- ]to[- ]toe|full figure|entire outfit|whole.*outfit|complete.*outfit|full ensemble", re.I)
BW_RE = re.compile(r"\b(monochrome|greyscale|grayscale|black and white|sepia)\b", re.I)
FEM_FOOT = ("heel", "pump", "slingback", "stiletto", "mule", "ballet", "wedge", "gladiator", "evening sandal")
FEM_ACC = ("earring", "headscarf", "clutch", "purse", "handbag", "tiara", "necklace")


def test_male_outfits_gender_gated():
    bad = [fc.outfit_caption("male", 0, i) for i in range(2000)
           if any(k in fc.outfit_caption("male", 0, i).lower() for k in FEM_FOOT + FEM_ACC)]
    assert not bad, bad[:3]
    # men get male garments only (no women's dresses/skirts/gowns/blouses).
    # NB: "dress shirt"/"dress shoes" are men's items -> require a standalone "dress".
    fem_re = re.compile(r"\b(skirt|gown|blouse|camisole|bodysuit|leggings|maxi dress|sundress)\b"
                        r"|\bdress\b(?!\s+(shirt|shoe|trouser|pant|code))", re.I)
    leak = [c for c in (fc.outfit_caption("male", 0, i) for i in range(1500)) if fem_re.search(c)]
    assert not leak, leak[:3]
    print("test_male_outfits_gender_gated OK")


def test_deepfashion_whitebg_and_tshirt():
    m = pp.augment("a woman wearing a white t - shirt and jeans in front of a white background", "deepfashion_1", cfg)
    f = m["final_prompt"]
    assert "white background" not in f.lower() and "white backdrop" not in f.lower(), f
    assert "t - shirt" not in f and "t-shirt" in f, f
    assert m["gender"] == "woman" and m["race"] is not None and m["race_injected"]
    assert "jeans" in f.lower()                  # outfit preserved
    print("test_deepfashion_whitebg_and_tshirt OK")


def test_fashion_full_length_color_adult():
    woman_cau = man_seen = 0
    for i in range(800):
        gender = "male" if i % 4 == 0 else "female"
        m = pp.augment(fc.outfit_caption(gender, 0, i), f"fsyn_{gender}_b0_{i}", cfg)
        f = m["final_prompt"]
        assert FULL_RE.search(f), f                 # whole outfit visible
        assert "color" in f.lower()                 # color enforced
        assert not BW_RE.search(f)                  # never monochrome
        assert m["age_band"] == "25-35"
        assert m["gender"] == ("man" if gender == "male" else "woman")  # gender preserved
        man_seen += gender == "male"
    assert man_seen > 0
    print("test_fashion_full_length_color_adult OK")


def test_fashion_amateur_and_dedup():
    am = sum(pp.augment(fc.outfit_caption("female", 0, i), f"a{i}", cfg)["is_amateur"] for i in range(1500))
    assert 0.05 < am / 1500 < 0.16, am / 1500
    g = pp.DiversityGuard()
    out = [pp.augment_diverse(fc.outfit_caption("female", 0, i), f"d{i}", cfg, guard=g)["final_prompt"] for i in range(300)]
    assert len(set(out)) >= 290, len(set(out))   # diverse, few collisions
    print(f"test_fashion_amateur_and_dedup OK (amateur={100*am/1500:.1f}%)")


ALL = [test_male_outfits_gender_gated, test_deepfashion_whitebg_and_tshirt,
       test_fashion_full_length_color_adult, test_fashion_amateur_and_dedup]

if __name__ == "__main__":
    for t in ALL:
        t()
    print(f"\nAll {len(ALL)} fashion_policy tests passed.")
