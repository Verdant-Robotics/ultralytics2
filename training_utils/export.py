from training_utils import (
    GiveModel
)
import os
import argparse
import torch
from ultralytics.utils import LOGGER


_EXPORT_SIZES = [
    ([2144, 768],  "_full_height"),
    ([2144, 4096], "_full_frame"),
    ([768, 768],   ""),
]


def Export(checkpoint_file_path):
    if os.path.exists(checkpoint_file_path):
        model = GiveModel(checkpoint_file_path)
    else:
        print(f"[ERROR] : Model {checkpoint_file_path} does not exists")
        exit(1)

    prefix = os.path.splitext(os.path.basename(checkpoint_file_path))[0]
    base_path, _ = os.path.split(checkpoint_file_path)

    for imgsz, suffix in _EXPORT_SIZES:
        path = model.export(format="onnx", imgsz=imgsz, opset=12)
        os.system(f"mv {path} {base_path}/{prefix}{suffix}.onnx")


def ExportCluster(checkpoint_file_path):
    """Export a cluster-trained model to ONNX with EmbeddingHead included as a second output.

    Produces one ONNX file per export size (matching Export), with output names:
      - output0:    standard YOLO pose detection output
      - embeddings: feature vectors at each grid point (B, embed_dim, n_grid_points)
    """
    import onnx
    from datetime import datetime
    from cluster_training import ClusterModelExportWrapper
    from ultralytics import __version__
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics.utils.export import torch2onnx

    if not os.path.exists(checkpoint_file_path):
        print(f"[ERROR] : Model {checkpoint_file_path} does not exist")
        exit(1)

    model = GiveModel(checkpoint_file_path)
    base = unwrap_model(model.model)

    assert hasattr(base, "embedding_head"), "Model does not have an embedding head."

    base.eval()
    detect_head = base.model[-1]
    detect_head.export = True
    detect_head.format = "onnx"

    from ultralytics.nn.modules import C2f
    for m in base.modules():
        if isinstance(m, C2f):
            m.forward = m.forward_split

    wrapper = ClusterModelExportWrapper(base)

    base_path, _ = os.path.split(checkpoint_file_path)
    prefix = os.path.splitext(os.path.basename(checkpoint_file_path))[0]

    inner_model = model.model
    yaml_dict = getattr(inner_model, "yaml", {}) or {}
    data = (model.overrides or {}).get("data", "")
    base_metadata = {
        "description": f"Ultralytics model{' trained on ' + data if data else ''}",
        "author": "Ultralytics",
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": int(max(inner_model.stride)),
        "task": model.task,
        "batch": 1,
        "names": model.names,
        "channels": yaml_dict.get("channels", 3),
    }
    if model.task == "pose":
        base_metadata["kpt_shape"] = base.model[-1].kpt_shape
        if hasattr(model, "kpt_names"):
            base_metadata["kpt_names"] = model.kpt_names

    for (h, w), suffix in [(imgsz, s) for imgsz, s in _EXPORT_SIZES]:
        im = torch.zeros(1, 3, h, w)
        out_path = f"{base_path}/{prefix}{suffix}_cluster.onnx"
        torch2onnx(
            wrapper,
            im,
            out_path,
            opset=12,
            input_names=["images"],
            output_names=["output0", "embeddings"],
        )
        metadata = {**base_metadata, "imgsz": [h, w]}
        model_onnx = onnx.load(out_path)
        try:
            import onnxslim
            model_onnx = onnxslim.slim(model_onnx)
        except Exception as e:
            LOGGER.warning(f"onnxslim simplifier failure: {e}")
        for k, v in metadata.items():
            prop = model_onnx.metadata_props.add()
            prop.key, prop.value = k, str(v)
        onnx.save(model_onnx, out_path)
        print(f"Exported cluster model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the model")
    parser.add_argument(
        "-m",
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model weight checkpoint; This model will be exported")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Export cluster model with embedding head")
    args = parser.parse_args()
    
    if args.cluster:
        ExportCluster(args.checkpoint_path)
    else:
        Export(args.checkpoint_path)
