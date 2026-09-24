"""Replace mask einsums with equivalent batched matrix products, preserving weights."""

from pathlib import Path

import onnx
from onnx import helper


def optimize(source: Path, destination: Path) -> int:
    model = onnx.load(source, load_external_data=False)
    nodes = []
    count = 0
    for node in model.graph.node:
        equation = next((a.s.decode().replace(" ", "") for a in node.attribute if a.name == "equation"), "")
        if node.op_type != "Einsum" or equation != "bqc,bchw->bqhw":
            nodes.append(node)
            continue
        prefix = f"lb_mask_{count}_"
        left, right = node.input
        # Flatten H,W, multiply (B,Q,C) @ (B,C,HW), then restore (B,Q,H,W).
        for name, values in [("flat", [0, 0, -1]), ("bq", [0, 1]), ("hw", [2, 3])]:
            model.graph.initializer.append(helper.make_tensor(prefix+name, onnx.TensorProto.INT64,
                                                              [len(values)], values))
        nodes.extend([
            helper.make_node("Reshape", [right, prefix+"flat"], [prefix+"pixels"], name=prefix+"flatten"),
            helper.make_node("MatMul", [left, prefix+"pixels"], [prefix+"product"], name=prefix+"matmul"),
            helper.make_node("Shape", [left], [prefix+"lshape"], name=prefix+"ls"),
            helper.make_node("Shape", [right], [prefix+"rshape"], name=prefix+"rs"),
            helper.make_node("Gather", [prefix+"lshape", prefix+"bq"], [prefix+"batch_query"], axis=0, name=prefix+"g1"),
            helper.make_node("Gather", [prefix+"rshape", prefix+"hw"], [prefix+"height_width"], axis=0, name=prefix+"g2"),
            helper.make_node("Concat", [prefix+"batch_query", prefix+"height_width"], [prefix+"shape"], axis=0, name=prefix+"cat"),
            helper.make_node("Reshape", [prefix+"product", prefix+"shape"], list(node.output), name=prefix+"restore"),
        ])
        count += 1
    if count == 0:
        raise ValueError("Expected mask Einsum nodes were not found")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.save_model(model, destination)
    onnx.checker.check_model(str(destination))
    return count


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "data/models/manuscript/mask2former_line_v0_prev"
    count = optimize(directory / "mask2former_line_v0_prev.onnx", directory / "mask2former_line_v0_prev.matmul.onnx")
    print(f"Replaced {count} Einsum nodes; original graph and external weights unchanged. Parity verification still required.")
