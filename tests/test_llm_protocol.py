import math

from labelbench.contracts import Annotation, ProviderResult, RunResult
from labelbench.llm_protocol import apply_review, compact_candidates


def candidate(identifier: str, x: float = 10) -> Annotation:
    return Annotation(id=identifier, label="text", score=0.9, bbox_xywh=[x, 10, 20, 20], provider="test")


def test_compact_polygons_are_bounded_without_boxes_and_ids_stable() -> None:
    a = candidate("long-original-id")
    a.polygon = [[50+20*math.cos(i/50), 50+20*math.sin(i/50)] for i in range(315)]
    run = RunResult(run_id="r", image_name="x", image_size=[100,100], consensus_score=0,
                    providers={"test": ProviderResult(provider="test", model="m", image_name="x",
                               image_size=[100,100], annotations=[a], elapsed_seconds=0)})
    groups, aliases = compact_candidates(run, ["test"])
    row = groups[0][1][0]
    assert len(row) == 4
    assert len(row[3]) <= 8
    assert all(0 <= v <= 1 for point in row[3] for v in point)
    assert aliases[row[0]].id == "long-original-id"
    assert len(a.polygon) == 315


def test_decisions_preserve_removed_and_uncertain_but_export_only_accepted() -> None:
    aliases = {f"a{i}": candidate(str(i)) for i in range(4)}
    result = apply_review({"keep":["a0"], "remove":["a1"], "uncertain":["a2"]}, aliases, [100,100])
    assert [a.id for a in result["refined_annotations"]] == ["0"]
    assert [a.attributes["review_status"] for a in result["review_annotations"]] == ["kept", "removed", "uncertain", "uncertain"]
    assert "review_status" not in aliases["a0"].attributes


def test_normalized_edit_merge_average_and_add() -> None:
    aliases = {"a0":candidate("0"), "a1":candidate("1", 20), "a2":candidate("2")}
    poly = [[.2,.3],[.6,.3],[.6,.4],[.2,.4]]
    result = apply_review({"edit":[["a2",poly]], "merge":[[["a0","a1"]]],
                           "add":[["text",poly]]}, aliases, [100,100])
    output = result["refined_annotations"]
    assert output[0].bbox_xywh == [15,10,20,20]
    assert output[1].bbox_xywh == [20,30,40,10]
    assert result["removed_ids"] == ["1"]
    assert output[-1].attributes["review_status"] == "added"


def test_bad_geometry_and_unknown_ids_do_not_destroy_candidates() -> None:
    result = apply_review({"edit":[["a0", [[20,30],[40,30],[40,50]]]],
                           "remove":["unknown"]}, {"a0":candidate("0")}, [100,100])
    assert result["invalid_decisions"] == 2
    assert result["review_annotations"][0].attributes["review_status"] == "uncertain"


def test_conflicting_decisions_are_not_accepted() -> None:
    result = apply_review({"keep":["a0"], "edit":[["a0",[[0,0],[1,0],[1,1]]]]},
                          {"a0":candidate("0")}, [100,100])
    assert not result["refined_annotations"]
    assert result["invalid_decisions"] == 1


def test_empty_reply_does_not_claim_review_success() -> None:
    result = apply_review({"keep":[]}, {"a0":candidate("0")}, [100,100])
    assert result["notes"]
    assert not result["refined_annotations"]
