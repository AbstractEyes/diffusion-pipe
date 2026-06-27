"""Unit tests for utils.subject_bucket (subject extraction + dampened repeats + manifest).

Runnable standalone (stdlib only):
    python test/test_subject_bucket.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.subject_bucket import (
    normalize_subject, caption_features, dampened_repeats, bucket_repeats,
    build_manifest, SubjectBucketConfig,
)


def test_normalize_subject():
    assert normalize_subject("Fire Truck") == "truck"
    assert normalize_subject("the Police Officers") == "officer"
    assert normalize_subject("women") == "woman"
    assert normalize_subject("a CAR") == "car"
    assert normalize_subject("buses") == "bus"
    assert normalize_subject("") is None and normalize_subject(None) is None
    # head_noun off keeps the phrase (singularized last word)
    assert normalize_subject("fire truck", head_noun=False) == "fire truck"
    print("test_normalize_subject OK")


def test_caption_features():
    vlm = json.dumps({"subjects": [
        {"name": "woman", "attributes": ["blonde hair", "1girl"]},
        {"name": "jeans", "attributes": ["blue"]},
    ], "actions": ["standing"], "setting": "unknown"})
    subj, attrs, sec = caption_features(vlm)
    assert subj == "woman", subj
    assert attrs == ("blonde_hair",), attrs        # '1girl' meta-tag dropped, spaces->_
    assert sec == "jean", sec                       # subjects[1] normalized + singularized
    # garbage / sentinels -> empty
    assert caption_features("__PARSEFAIL__") == (None, (), None)
    assert caption_features("{not json") == (None, (), None)
    assert caption_features("") == (None, (), None)
    print("test_caption_features OK")


def test_dampened_repeats():
    # images == top -> 1x
    assert dampened_repeats(10000, 10000) == 1
    # sqrt damping (alpha=0.5): sqrt(top/images), then capped at max_repeats
    assert dampened_repeats(2500, 10000, alpha=0.5) == 2        # sqrt(4)=2
    assert dampened_repeats(100, 10000, alpha=0.5, max_repeats=8) == 8   # sqrt(100)=10 -> cap 8
    assert dampened_repeats(100, 10000, alpha=0.5, max_repeats=50) == 10
    # alpha=1 -> no balancing
    assert dampened_repeats(1, 10000, alpha=1.0) == 1
    # alpha=0 -> equalize (but still capped)
    assert dampened_repeats(100, 10000, alpha=0.0, max_repeats=8) == 8   # 100x -> cap 8
    assert dampened_repeats(2000, 10000, alpha=0.0, max_repeats=50) == 5 # 5x
    # effective-samples ceiling: rep can't push images*rep past cap_mult*top
    assert dampened_repeats(0, 10) == 1
    # monotonic: smaller buckets get >= repeats
    for a, b in [(50, 500), (500, 5000)]:
        assert dampened_repeats(a, 10000) >= dampened_repeats(b, 10000)
    print("test_dampened_repeats OK")


def test_bucket_repeats():
    sizes = {"big": 10000, "mid": 1000, "small": 100}
    reps = bucket_repeats(sizes, SubjectBucketConfig(balance_alpha=0.5, max_repeats=8))
    assert reps["big"] == 1
    assert reps["mid"] == 3        # round(sqrt(10)) = round(3.16) = 3
    assert reps["small"] == 8      # sqrt(100)=10 -> cap 8
    print("test_bucket_repeats OK")


def _row(name, attrs=None):
    return json.dumps({"subjects": [{"name": name, "attributes": attrs or []}],
                       "actions": [], "setting": "unknown"})


def test_build_manifest():
    rows = []
    # 60 women, 30 men, 12 cars (all >= min_bucket_size=10 -> own buckets), 3 rare truck (-> misc)
    rows += [(f"w{i}", _row("woman")) for i in range(60)]
    rows += [(f"m{i}", _row("man")) for i in range(30)]
    rows += [(f"c{i}", _row("car")) for i in range(12)]
    rows += [(f"t{i}", _row("trucks")) for i in range(3)]   # 'trucks'->'truck', small
    rows += [(f"n{i}", "__NO_TAGS__") for i in range(2)]    # no subject

    cfg = SubjectBucketConfig(balance_alpha=0.5, max_repeats=8, min_bucket_size=10,
                              use_semantic=False, split_oversized=False)
    man = build_manifest(rows, cfg)
    R = man["rows"]
    # top = 60 (woman). woman 1x; car (12) gets a lift; no-subject rows -> default 1
    assert R["w0"]["bucket"] == "woman" and R["w0"]["num_repeats"] == 1
    assert R["m0"]["bucket"] == "man"
    assert R["c0"]["bucket"] == "car" and R["c0"]["num_repeats"] >= 2   # sqrt(60/12)=2.24 -> 2+
    assert R["t0"]["bucket"] in ("misc", "woman", "man", "car")          # small -> merged or misc
    assert R["n0"]["bucket"] == "misc_no_subject" and R["n0"]["num_repeats"] == 1
    # effective samples of the small/car bucket should be bounded (no runaway)
    car_eff = 12 * R["c0"]["num_repeats"]
    assert car_eff <= 1.25 * 60 + 1
    print(f"test_build_manifest OK (car {R['c0']['num_repeats']}x, buckets={man['num_buckets']})")


def test_oversized_split():
    # one giant 'woman' bucket with differentiating attributes -> should split
    rows = []
    for i in range(600):
        attr = "blonde_hair" if i % 2 == 0 else "red_dress"
        rows.append((f"w{i}", _row("woman", [attr])))
    cfg = SubjectBucketConfig(use_semantic=False, split_oversized=True, max_bucket_size=250)
    man = build_manifest(rows, cfg)
    buckets = set(r["bucket"] for r in man["rows"].values())
    assert len(buckets) >= 2, buckets        # 600 > 250 -> split by attribute
    assert any("woman" in b and "." in b for b in buckets), buckets
    print(f"test_oversized_split OK (buckets={sorted(buckets)[:4]}...)")


ALL = [test_normalize_subject, test_caption_features, test_dampened_repeats,
       test_bucket_repeats, test_build_manifest, test_oversized_split]

if __name__ == "__main__":
    for t in ALL:
        t()
    print(f"\nAll {len(ALL)} subject_bucket tests passed.")
