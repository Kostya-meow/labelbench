import json

from labelbench.llm_batches import MAX_BATCH_CHARS, collect_reviews, split_groups


def test_batches_preserve_groups_and_bound_context() -> None:
    groups = [[index, [["model", "text", 0.9, "x" * 1600]]] for index in range(100)]
    batches = split_groups(groups)
    assert len(batches) > 1
    assert [group for batch in batches for group in batch] == groups
    assert all(len(json.dumps(batch, separators=(",", ":"))) <= MAX_BATCH_CHARS for batch in batches)
    assert max(map(len, split_groups([[i, []] for i in range(101)]))) <= 40


def test_overflow_splits_without_losing_groups() -> None:
    calls: list[int] = []

    def request(batch: list) -> dict:
        calls.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("exceed_context_size_error")
        return {"parsed": True, "decisions": {"pick": [[row[0], 0] for row in batch]},
                "usage": {"total_tokens": 7}}

    result = collect_reviews([[i, []] for i in range(8)], request)
    assert calls == [8, 4, 2, 2, 4, 2, 2]
    assert result["complete"]
    assert result["decisions"]["pick"] == [[i, 0] for i in range(8)]
    assert result["usage"]["total_tokens"] == 28


def test_partial_failure_retains_success_and_marks_remaining_uncertain() -> None:
    def request(batch: list) -> dict:
        if batch[0][0] >= 40:
            raise RuntimeError("LM Studio unavailable")
        return {"parsed": True, "decisions": {"drop": [0]}}

    result = collect_reviews([[i, []] for i in range(81)], request)
    assert result["parsed"] and not result["complete"]
    assert result["decisions"]["drop"] == [0]
    assert sorted(result["decisions"]["uncertain"]) == list(range(40, 81))
    assert "LM Studio unavailable" in result["decisions"]["notes"]


def test_invalid_json_never_implicitly_accepts_singletons() -> None:
    result = collect_reviews([[0, []], [1, []]], lambda batch: {"parsed": False})
    assert not result["parsed"] and not result["complete"]
    assert result["decisions"]["uncertain"] == [0, 1]


def test_batch_cannot_change_unseen_objects() -> None:
    result = collect_reviews([[0, []]], lambda batch: {
        "parsed": True, "decisions": {"drop": [200], "pick": [[100, 0], [0, 1]]},
    })
    assert result["decisions"]["drop"] == []
    assert result["decisions"]["pick"] == [[0, 1]]
    assert result["ignored_decisions"] == 2
