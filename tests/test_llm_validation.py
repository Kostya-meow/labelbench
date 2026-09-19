import pytest

from labelbench.llm_validation import decision_errors


@pytest.mark.parametrize("decisions", [
    {"pick": [[7, 0], [7, 1]]},
    {"pick": [[7, 0]], "drop": [7]},
    {"pick": [[7, 99]]},
    {"pick": [[7, True]]},
    {"pick": [[99, 0]]},
    {"pick": []},
    {"fuse": [[7, [[0, 0], [0, 0], [0, 0]]]]},
])
def test_bad_decisions_are_rejected(decisions: dict) -> None:
    assert decision_errors(decisions, [[7, [[], []]]])


def test_one_choice_or_uncertain_is_valid() -> None:
    groups = [[7, [[], []]], [8, [[]]]]
    assert not decision_errors({"pick": [[7, 1], [8, 0]]}, groups)
    assert not decision_errors({"uncertain": [7], "pick": [[8, 0]]}, groups)
    assert decision_errors({"pick": []}, [[8, [[]]]])
