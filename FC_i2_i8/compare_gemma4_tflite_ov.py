"""
Compare TFLite vs OpenVINO model outputs for the Gemma4 i2 model.

The model is a stateless decode step with 38 inputs (KV caches, positional
embeddings, attention masks, token embeddings) and 31 outputs (updated KV
caches + logits).

Primary comparison: StatefulPartitionedCall:30  shape=[1,1,262144]  float32
KV cache outputs: TFLite produces int8; OV produces float32 which is quantized
back to int8 using TFLite's per-tensor scale/zero_point for an int8 vs int8 comparison.

Usage:
    python3 compare_gemma4_tflite_ov.py
    python3 compare_gemma4_tflite_ov.py --tflite-only
    python3 compare_gemma4_tflite_ov.py --ov-only [--device CPU]
    python3 compare_gemma4_tflite_ov.py --seed 123 --top-k 20
"""

import argparse
import numpy as np

TFLITE_MODEL = "gemma4_model.tflite"
OV_XML = "gemma4_i2.xml"

# The primary output: logits over the full vocabulary
LOGITS_OUTPUT = "StatefulPartitionedCall:30"


def build_inputs(tfl_details: list, rng: np.random.Generator) -> dict:
    """
    Build a dict {name: np.ndarray} matched to TFLite input dtype/shape.
    Attention masks are set to 0.0 (attend all positions).
    """
    inputs = {}
    for d in tfl_details:
        name = d["name"]
        shape = tuple(d["shape"])
        dtype = d["dtype"]

        if "mask" in name:
            # Attention mask: 0 = attend, large negative = mask out.
            # Use 0 (attend all) for a simple functional comparison.
            inputs[name] = np.zeros(shape, dtype=np.float32)
        elif dtype == np.float32:
            inputs[name] = rng.standard_normal(shape).astype(np.float32)
        elif dtype == np.int8:
            inputs[name] = rng.integers(-128, 128, size=shape, dtype=np.int8)
        else:
            raise ValueError(f"Unexpected dtype {dtype} for input {name}")
    return inputs


def run_tflite(inputs: dict) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=TFLITE_MODEL)
    interp.allocate_tensors()

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    for d in inp_details:
        interp.set_tensor(d["index"], inputs[d["name"]])

    interp.invoke()

    results = {}
    for d in out_details:
        results[d["name"]] = {
            "data": interp.get_tensor(d["index"]),
            "scale": d.get("quantization_parameters", {}).get("scales", np.array([])),
            "zero_point": d.get("quantization_parameters", {}).get(
                "zero_points", np.array([])
            ),
        }
    return results


def run_openvino(inputs: dict, device: str) -> dict:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(OV_XML)
    compiled = core.compile_model(model, device)
    infer_req = compiled.create_infer_request()

    # OV inputs have the same names as TFLite
    infer_req.infer(inputs)

    results = {}
    for out in compiled.outputs:
        name = out.any_name
        results[name] = infer_req.get_tensor(out).data.copy()
    return results


def dequantize(data: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    if scale.size == 0:
        return data.astype(np.float32)
    return (data.astype(np.float32) - float(zero_point[0])) * float(scale[0])


def compare_logits(tfl_logits: np.ndarray, ov_logits: np.ndarray, top_k: int) -> None:
    tfl = tfl_logits.flatten().astype(np.float32)
    ov_ = ov_logits.flatten().astype(np.float32)

    diff = np.abs(tfl - ov_)
    cos_sim = float(
        np.dot(tfl, ov_) / (np.linalg.norm(tfl) * np.linalg.norm(ov_) + 1e-12)
    )

    print(f"\n{'='*60}")
    print(f"  Logits shape     : {tfl_logits.shape}")
    print(f"  Max |diff|       : {diff.max():.6f}")
    print(f"  Mean |diff|      : {diff.mean():.6f}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # Top-k token agreement
    tfl_topk = np.argsort(tfl)[-top_k:][::-1]
    ov_topk = np.argsort(ov_)[-top_k:][::-1]
    overlap = len(set(tfl_topk) & set(ov_topk))
    print(f"  Top-{top_k} token overlap: {overlap}/{top_k}")
    print(f"  Argmax TFLite    : {tfl_topk[0]}  (logit={tfl[tfl_topk[0]]:.4f})")
    print(f"  Argmax OV        : {ov_topk[0]}  (logit={ov_[ov_topk[0]]:.4f})")
    print(f"{'='*60}\n")


def quantize(data: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
    """Quantize float32 → int8 using TFLite's per-tensor scale/zero_point."""
    q = np.round(data.astype(np.float32) / float(scale[0])) + float(zero_point[0])
    return np.clip(q, -128, 127).astype(np.int8)


def compare_kv_caches(tfl_outputs: dict, ov_outputs: dict) -> None:
    """
    Quantize OV float32 KV cache outputs to int8 using TFLite's quantization
    parameters, then compare int8 vs int8.
    """
    names = sorted(k for k in tfl_outputs if k != LOGITS_OUTPUT)
    max_diffs = []
    mean_diffs = []
    exact_matches = []
    total_elems = []
    cos_sims = []

    for name in names:
        tfl_entry = tfl_outputs[name]
        scale = tfl_entry["scale"]
        zero_point = tfl_entry["zero_point"]

        tfl_i8 = tfl_entry["data"].flatten().astype(np.int32)

        # Quantize OV float32 → int8 using TFLite's scale/zero_point
        ov_i8 = quantize(ov_outputs[name], scale, zero_point).flatten().astype(np.int32)

        diff = np.abs(tfl_i8 - ov_i8)
        max_diffs.append(diff.max())
        mean_diffs.append(diff.mean())
        exact_matches.append((tfl_i8 == ov_i8).sum())
        total_elems.append(tfl_i8.size)

        tfl_f = tfl_i8.astype(np.float32)
        ov_f  = ov_i8.astype(np.float32)
        cos = float(
            np.dot(tfl_f, ov_f)
            / (np.linalg.norm(tfl_f) * np.linalg.norm(ov_f) + 1e-12)
        )
        cos_sims.append(cos)

    total = sum(total_elems)
    exact = sum(exact_matches)
    print(f"KV cache outputs ({len(names)} tensors, int8 vs quantized-OV int8):")
    print(f"  Max |diff|     — mean: {np.mean(max_diffs):.2f}   max: {np.max(max_diffs):.2f}")
    print(f"  Mean |diff|    — mean: {np.mean(mean_diffs):.4f}")
    print(f"  Exact matches  : {exact} / {total}  ({100*exact/total:.1f}%)")
    print(f"  Cosine sim     — mean: {np.mean(cos_sims):.6f}   min: {np.min(cos_sims):.6f}")


def print_single_output(outputs, label: str) -> None:
    """Print a brief summary when running only one backend."""
    if isinstance(outputs, dict):
        logits = outputs.get(LOGITS_OUTPUT)
        if logits is not None:
            if isinstance(logits, dict):
                logits = logits["data"].astype(np.float32)
            logits = logits.flatten()
            top5 = np.argsort(logits)[-5:][::-1]
            print(f"\n[{label}] Logits shape: {logits.shape}")
            print(f"  Argmax token : {top5[0]}  (logit={logits[top5[0]]:.4f})")
            print(f"  Top-5 tokens : {top5.tolist()}")
            print(f"  Logits range : [{logits.min():.4f}, {logits.max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="CPU", help="OpenVINO device (CPU, NPU, …)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K tokens to compare")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tflite-only", action="store_true", help="Run TFLite inference only")
    mode.add_argument("--ov-only", action="store_true", help="Run OpenVINO inference only")
    args = parser.parse_args()

    run_tfl = not args.ov_only
    run_ov = not args.tflite_only

    # Build inputs from TFLite metadata (same names/shapes for OV)
    from ai_edge_litert.interpreter import Interpreter as _Interp
    _interp = _Interp(model_path=TFLITE_MODEL)
    _interp.allocate_tensors()
    inp_details = _interp.get_input_details()

    rng = np.random.default_rng(args.seed)
    inputs = build_inputs(inp_details, rng)

    print(f"Inputs : {len(inputs)} tensors")
    print(f"Seed   : {args.seed}")

    tfl_outputs = None
    ov_outputs = None

    if run_tfl:
        print("\n[TFLite] Running inference...")
        tfl_outputs = run_tflite(inputs)
        logits = tfl_outputs[LOGITS_OUTPUT]["data"]
        print(f"  Logits shape : {logits.shape}  dtype: {logits.dtype}")
        print(f"  Logits range : [{logits.min():.4f}, {logits.max():.4f}]")

    if run_ov:
        print(f"\n[OpenVINO ({args.device})] Running inference...")
        ov_outputs = run_openvino(inputs, args.device)
        logits = ov_outputs[LOGITS_OUTPUT]
        print(f"  Logits shape : {logits.shape}  dtype: {logits.dtype}")
        print(f"  Logits range : [{logits.min():.4f}, {logits.max():.4f}]")

    if run_tfl and run_ov:
        tfl_logits = tfl_outputs[LOGITS_OUTPUT]["data"].astype(np.float32)
        ov_logits = ov_outputs[LOGITS_OUTPUT].astype(np.float32)
        print("\n--- Logits comparison ---")
        compare_logits(tfl_logits, ov_logits, args.top_k)
        print("--- KV cache comparison (dequantized TFLite vs OV float32) ---")
        compare_kv_caches(tfl_outputs, ov_outputs)
    elif run_tfl:
        print_single_output(tfl_outputs, "TFLite")
    else:
        print_single_output(ov_outputs, "OV")


if __name__ == "__main__":
    main()
