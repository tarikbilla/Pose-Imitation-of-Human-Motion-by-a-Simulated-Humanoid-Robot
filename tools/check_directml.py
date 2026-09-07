import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

MATRIX = 1024
DEPTH = 8
WARMUP = 3
RUNS = 20


def build_model(path):
    from onnx import TensorProto, helper, numpy_helper

    nodes = []
    initializers = []
    current = "input"
    rng = np.random.default_rng(0)
    for index in range(DEPTH):
        weight_name = f"w{index}"
        weight = rng.standard_normal((MATRIX, MATRIX)).astype(np.float32) * 0.02
        initializers.append(numpy_helper.from_array(weight, weight_name))
        output = "output" if index == DEPTH - 1 else f"h{index}"
        nodes.append(helper.make_node("MatMul", [current, weight_name], [output]))
        current = output

    graph = helper.make_graph(
        nodes,
        "matmul_chain",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, MATRIX, MATRIX])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, MATRIX, MATRIX])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    with open(path, "wb") as handle:
        handle.write(model.SerializeToString())


def benchmark(model_path, provider):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        session = ort.InferenceSession(model_path, options, providers=[provider])
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    if provider not in session.get_providers():
        return {"available": False, "error": f"provider not applied: {session.get_providers()}"}

    data = np.random.default_rng(1).standard_normal((1, MATRIX, MATRIX)).astype(np.float32)
    for _ in range(WARMUP):
        session.run(None, {"input": data})

    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        session.run(None, {"input": data})
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    flops = 2.0 * DEPTH * MATRIX ** 3
    median = timings[len(timings) // 2]
    return {
        "available": True,
        "providers": session.get_providers(),
        "median_ms": median,
        "min_ms": timings[0],
        "gflops": flops / (median / 1000.0) / 1e9,
    }


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "_bench_matmul.onnx")

    print("=" * 62)
    print("M0 / E3  ONNX Runtime execution provider check")
    print("=" * 62)
    print(f"onnxruntime      {ort.__version__}")
    available = ort.get_available_providers()
    print(f"providers        {available}")

    if "DmlExecutionProvider" not in available:
        print()
        print("FATAL: DmlExecutionProvider missing. Wrong package installed?")
        print("       Need 'onnxruntime-directml', not 'onnxruntime'.")
        return 1

    print(f"building benchmark model ({DEPTH} chained {MATRIX}x{MATRIX} MatMul)...")
    build_model(model_path)
    print(f"model size       {os.path.getsize(model_path) / 1e6:.1f} MB")
    print("-" * 62)

    result = {"onnxruntime_version": ort.__version__, "available_providers": available}
    for provider in ("DmlExecutionProvider", "CPUExecutionProvider"):
        print(f"benchmarking {provider} ...")
        measured = benchmark(model_path, provider)
        result[provider] = measured
        if measured["available"]:
            print(f"    median {measured['median_ms']:8.2f} ms    {measured['gflops']:7.1f} GFLOP/s")
        else:
            print(f"    unavailable: {measured['error']}")

    print("-" * 62)
    dml = result.get("DmlExecutionProvider", {})
    cpu = result.get("CPUExecutionProvider", {})
    verdict_ok = False
    if dml.get("available") and cpu.get("available"):
        speedup = cpu["median_ms"] / dml["median_ms"]
        result["dml_speedup_over_cpu"] = speedup
        print(f"DirectML speedup over CPU   {speedup:.1f}x")
        verdict_ok = speedup > 2.0
        if verdict_ok:
            print("VERDICT: DirectML is doing real GPU work. GPU path is viable.")
        else:
            print("VERDICT: no meaningful speedup - DirectML may be falling back.")
    result["gpu_path_viable"] = verdict_ok
    print("=" * 62)

    out_path = os.path.join(out_dir, "m0_e3_directml.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"written: {out_path}")

    try:
        os.remove(model_path)
    except OSError:
        pass

    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
