from training_utils import (
    GiveModel
)
import os
import argparse


def Export(checkpoint_file_path, num_protos=3):
    if os.path.exists(checkpoint_file_path):
        model = GiveModel(checkpoint_file_path)
    else:
        print(f"[ERROR] : Model {checkpoint_file_path} does not exists")
        exit(1)

    prefix = os.path.splitext(os.path.basename(checkpoint_file_path))[0]

    base_path, _ = os.path.split(checkpoint_file_path)
    path = model.export(format="onnx", imgsz=[2144, 768], opset=12)
    os.system(f"mv {path} {base_path}/{prefix}_full_height.onnx")

    path = model.export(format="onnx", imgsz=[2144, 4096], opset=12)
    os.system(f"mv {path} {base_path}/{prefix}_full_frame.onnx")

    path = model.export(format="onnx", imgsz=[2144, 320], opset=12)
    os.system(f"mv {path} {base_path}/{prefix}_full_height_narrow.onnx")

    path = model.export(format="onnx", imgsz=[768, 768], opset=12)

    # Pose-prompt only: the example-conditioned ("what-if") ABC model. It runs on cached per-box
    # embeddings + example prototypes (no image), so there are no image-size variants - one file.
    # opset 16 (the transformer's MultiheadAttention needs >= 13; runs under ONNX Runtime / TRT <= 16).
    if hasattr(model.model, "export_abc_onnx"):
        out = f"{base_path}/{prefix}_prompt.onnx"
        model.model.export_abc_onnx(out, num_protos=num_protos, opset=16)
        print(f"Exported what-if ABC model to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the model")
    parser.add_argument(
        "-m",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model weight checkpoint; This model will be exported")
    parser.add_argument(
        "--num-protos",
        type=int,
        default=3,
        help="pose-prompt only: number of class prototype slots baked into the what-if ABC model "
             "(the max number of A/B/C classes; drop a class at runtime with the empty sentinel)")

    args = parser.parse_args()
    Export(args.checkpoint_path, num_protos=args.num_protos)
