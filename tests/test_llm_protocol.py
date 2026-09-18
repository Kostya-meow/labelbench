import math

from labelbench.contracts import Annotation, ProviderResult, RunResult
from labelbench.llm_protocol import apply_review, compact_candidates


def candidate(
    identifier: str,
    x: float = 10,
    y: float = 10,
    provider: str = "model_a",
    points: int = 4,
) -> Annotation:
    polygon = [[x, y], [x + 20, y], [x + 20, y + 10], [x, y + 10]]
    if points > 4:
        polygon = [
            [x + 10 + 10 * math.cos(index * 2 * math.pi / points),
             y + 5 + 5 * math.sin(index * 2 * math.pi / points)]
            for index in range(points)
        ]
    return Annotation(
        id=identifier,
        label="text_line",
        score=0.9,
        bbox_xywh=[x, y, 20, 10],
        polygon=polygon,
        provider=provider,
    )


def run_with(*annotations: Annotation) -> RunResult:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.provider, []).append(annotation)
    providers = {
        provider: ProviderResult(
            provider=provider,
            model=provider,
            image_name="x.jpg",
            image_size=[100, 100],
            annotations=items,
            elapsed_seconds=0,
        )
        for provider, items in grouped.items()
    }
    return RunResult(
        run_id="r",
        image_name="x.jpg",
        image_size=[100, 100],
        consensus_score=0,
        providers=providers,
    )


def test_payload_has_only_polygons_and_simplifies_to_fifty_points() -> None:
    source = candidate("long", points=315)
    payload, aliases, groups = compact_candidates(run_with(source), ["model_a"])
    row = payload[0][1][0]
    assert len(row) == 4
    assert len(row[3]) == 50
    assert all(0 <= coordinate <= 1 for point in row[3] for coordinate in point)
    assert "bbox" not in str(payload)
    assert next(iter(aliases.values())).id == "long"
    assert len(source.polygon or []) == 315
    assert len(next(iter(groups.values()))) == 1


def test_overlapping_models_form_one_group_but_separate_lines_do_not() -> None:
    payload, _, groups = compact_candidates(
        run_with(
            candidate("a", provider="model_a"),
            candidate("b", x=11, y=11, provider="model_b"),
            candidate("c", y=40, provider="model_b"),
        ),
        ["model_a", "model_b"],
    )
    assert sorted(len(members) for members in groups.values()) == [1, 2]
    assert sorted(len(rows) for _, rows in payload) == [1, 2]


def test_pick_keeps_exactly_one_candidate_from_duplicate_group() -> None:
    _, aliases, groups = compact_candidates(
        run_with(candidate("a"), candidate("b", x=11, provider="model_b")),
        ["model_a", "model_b"],
    )
    group_id, _members = next(iter(groups.items()))
    result = apply_review(
        {"pick": [[group_id, 1]]},
        aliases,
        groups,
        [100, 100],
    )
    assert [item.id for item in result["refined_annotations"]] == ["b"]
    assert [item.attributes["review_status"] for item in result["review_annotations"]] == [
        "removed",
        "kept",
    ]


def test_unresolved_duplicate_group_never_becomes_green() -> None:
    _, aliases, groups = compact_candidates(
        run_with(candidate("a"), candidate("b", x=11, provider="model_b")),
        ["model_a", "model_b"],
    )
    result = apply_review({}, aliases, groups, [100, 100])
    assert result["refined_annotations"] == []
    assert result["unresolved_groups"] == list(groups)
    assert all(
        item.attributes["review_status"] == "uncertain"
        for item in result["review_annotations"]
    )


def test_invalid_local_choice_keeps_duplicate_group_unresolved() -> None:
    _, aliases, groups = compact_candidates(
        run_with(candidate("a"), candidate("b", x=11, provider="model_b")),
        ["model_a", "model_b"],
    )
    group_id = next(iter(groups))
    result = apply_review({"pick": [[group_id, 99]]}, aliases, groups, [100, 100])
    assert result["invalid_decisions"] == 1
    assert result["unresolved_groups"] == [group_id]
    assert result["refined_annotations"] == []


def test_fuse_outputs_one_polygon_and_removes_original_candidates() -> None:
    _, aliases, groups = compact_candidates(
        run_with(candidate("a"), candidate("b", x=11, provider="model_b")),
        ["model_a", "model_b"],
    )
    group_id = next(iter(groups))
    polygon = [[0.1, 0.1], [0.32, 0.1], [0.32, 0.21], [0.1, 0.21]]
    result = apply_review(
        {"fuse": [[group_id, polygon]]}, aliases, groups, [100, 100]
    )
    assert len(result["refined_annotations"]) == 1
    merged = result["refined_annotations"][0]
    assert merged.polygon == [[10, 10], [32, 10], [32, 21], [10, 21]]
    assert merged.attributes["review_status"] == "merged"
    assert len(result["removed_ids"]) == 2


def test_singletons_are_kept_unless_explicitly_rejected() -> None:
    _, aliases, groups = compact_candidates(run_with(candidate("a")), ["model_a"])
    accepted = apply_review({}, aliases, groups, [100, 100])
    rejected = apply_review({"drop": [next(iter(groups))]}, aliases, groups, [100, 100])
    assert [item.id for item in accepted["refined_annotations"]] == ["a"]
    assert rejected["refined_annotations"] == []


def test_polygonless_candidates_are_not_sent_to_vlm() -> None:
    source = candidate("box-only")
    source.polygon = None
    payload, aliases, groups = compact_candidates(run_with(source), ["model_a"])
    assert payload == []
    assert aliases == {}
    assert groups == {}
