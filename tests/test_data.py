import pytest

from ticketnlp.data import assert_no_leakage, duplicate_groups, normalized


def test_exact_and_near_duplicates_share_component():
    groups, audit = duplicate_groups(
        [
            "Where is my card?",
            "WHERE is my card!",
            "My transfer is pending",
            "I would like to cancel my bank transfer please",
            "I would like to cancel my bank transfer please!",
        ],
        0.9,
    )
    assert groups[0] == groups[1]
    assert groups[3] == groups[4]
    assert groups[0] != groups[2]
    assert audit["normalized_duplicate_rows"] == 2


def test_near_duplicate_punctuation_independent():
    groups, _ = duplicate_groups(
        [
            "I have been waiting for my new replacement card for two weeks now",
            "I have been waiting for my new replacement card for two weeks now please",
        ],
        0.9,
    )
    assert groups[0] == groups[1]


def test_group_leakage_detected():
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage({"train": [{"group": 1}], "test": [{"group": 1}]})
    assert_no_leakage({"train": [{"group": 1}], "test": [{"group": 2}]})


def test_empty_normalization_rejected():
    with pytest.raises(ValueError, match="empty"):
        duplicate_groups(["!!!"])
    assert normalized("  CARD, arrived! ") == "card arrived"
