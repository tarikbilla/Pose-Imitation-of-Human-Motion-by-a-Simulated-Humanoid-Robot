import argparse
import os
import re
import subprocess
import sys
import urllib.request

A3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(A3_ROOT, "models")
WORK_DIR = os.path.join(MODEL_DIR, "_motionbert_src")

BASE = "https://huggingface.co/walterzhu/MotionBERT/resolve/main"
SOURCES = (
    "lib/model/DSTformer.py",
    "lib/model/drop.py",
)
VARIANTS = {
    "lite": {
        "checkpoint": "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        "dim_feat": 256,
        "mlp_ratio": 4,
        "output": "motionbert_lite.onnx",
    },
    "full": {
        "checkpoint": "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin",
        "dim_feat": 512,
        "mlp_ratio": 2,
        "output": "motionbert_full.onnx",
    },
}

# DirectML has no MatMul for 5-D tensors. The temporal attention builds
# (B, H, N, T, C) operands; folding the leading axes into one keeps the maths
# identical and makes the graph run on the GPU.
PATCH_FROM = """        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt #(B, H, N, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
        return x"""

PATCH_TO = """        Bq, Hq, Nq, Tq, Cq = qt.shape
        q3 = qt.reshape(Bq * Hq * Nq, Tq, Cq)
        k3 = kt.reshape(Bq * Hq * Nq, Tq, Cq)
        v3 = vt.reshape(Bq * Hq * Nq, Tq, Cq)

        attn = (q3 @ k3.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v3
        x = x.reshape(Bq, Hq, Nq, Tq, Cq)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
        return x"""


def fetch(relative, target):
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"  downloading {relative}")
    urllib.request.urlretrieve(f"{BASE}/{relative}", target)
    return True


def prepare_sources():
    for relative in SOURCES:
        fetch(relative, os.path.join(WORK_DIR, relative))
    for package in ("lib", "lib/model"):
        init = os.path.join(WORK_DIR, package, "__init__.py")
        os.makedirs(os.path.dirname(init), exist_ok=True)
        if not os.path.isfile(init):
            open(init, "w").close()

    path = os.path.join(WORK_DIR, "lib", "model", "DSTformer.py")
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    if PATCH_TO in source:
        print("  DSTformer already patched")
        return
    if PATCH_FROM not in source:
        raise RuntimeError("upstream DSTformer changed; patch no longer applies")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source.replace(PATCH_FROM, PATCH_TO))
    print("  patched temporal attention to 3-D matmuls")


def export(variant, sequence):
    import torch

    spec = VARIANTS[variant]
    sys.path.insert(0, WORK_DIR)
    from lib.model.DSTformer import DSTformer

    checkpoint = os.path.join(WORK_DIR, "checkpoint", f"{variant}.bin")
    fetch(spec["checkpoint"], checkpoint)

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=spec["dim_feat"], dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=spec["mlp_ratio"],
        num_joints=17, maxlen=243, att_fuse=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("model_pos", state.get("model", state))
    if not isinstance(weights, dict):
        weights = weights.state_dict()
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"  WARNING missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    dummy = torch.zeros(1, sequence, 17, 3, dtype=torch.float32)
    output = os.path.join(MODEL_DIR, spec["output"])
    torch.onnx.export(
        model, dummy, output,
        input_names=["keypoints2d"], output_names=["pose3d"],
        dynamic_axes={"keypoints2d": {0: "batch", 1: "frames"},
                      "pose3d": {0: "batch", 1: "frames"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  wrote {output}  ({os.path.getsize(output) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download MotionBERT and export a DirectML-compatible ONNX lifter."
    )
    parser.add_argument("--variant", default="lite", choices=sorted(VARIANTS))
    parser.add_argument("--sequence", type=int, default=27)
    args = parser.parse_args()

    print("=" * 62)
    print(f"A3 / building the 3-D lifter  ({args.variant}, window {args.sequence})")
    print("=" * 62)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is required for the one-off conversion:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 1

    os.makedirs(MODEL_DIR, exist_ok=True)
    prepare_sources()
    export(args.variant, args.sequence)
    print("=" * 62)
    print("Verify with:  tools\\check_lifter.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
