from labelbench.contracts import Annotation
from labelbench.llm import apply_actions


def test_apply_llm_actions_keeps_valid_contract_and_bounds() -> None:
    annotations = [
        Annotation(
            id="line-1",
            label="text_line",
            score=0.8,
            bbox_xywh=[10, 10, 40, 12],
            provider="ppocr",
        ),
        Annotation(
            id="line-2",
            label="text_line",
            score=0.7,
            bbox_xywh=[10, 40, 40, 12],
            provider="docufcn",
        ),
    ]
    refined, removed, notes = apply_actions(
        {
            "actions": [
                {"action": "modify", "id": "line-1", "bbox_xywh": [90, 90, 30, 30]},
                {"action": "remove", "id": "line-2"},
                {"action": "add", "label": "text_line", "bbox_xywh": [0, 0, 20, 20]},
            ],
            "notes": "checked",
        },
        annotations,
        [100, 100],
    )

    assert removed == ["line-2"]
    assert notes == "checked"
    assert len(refined) == 2
    assert refined[0].bbox_xywh == [90.0, 90.0, 10.0, 10.0]
    assert refined[0].provider == "llm_review"


def test_apply_llm_actions_clamps_polygon_and_skips_invalid_polygon() -> None:
    annotations, _, _ = apply_actions(
        {
            "annotations": [
                {
                    "id": "line-1",
                    "label": "text_line",
                    "score": 0.9,
                    "bbox_xywh": [-5, -4, 30, 30],
                    "polygon": [[-5, -4], [30, 0], [30, 40]],
                },
                {
                    "id": "line-2",
                    "label": "text_line",
                    "score": 0.9,
                    "bbox_xywh": [1, 1, 10, 10],
                    "polygon": [[1, 1]],
                },
            ]
        },
        [],
        [20, 20],
    )

    assert annotations[0].polygon == [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]
    assert annotations[1].polygon is None
