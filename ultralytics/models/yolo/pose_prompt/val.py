# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.data.dataset import PosePromptDataset
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import colorstr


class PosePromptValidator(PoseValidator):
    """Validator for the "pose-prompt" task.

    Inference decoding is identical to Pose (embeddings ride in the auxiliary tuple and are ignored
    by NMS), so pose/keypoint metrics are reused unchanged. The dataset is a PosePromptDataset so the
    per-box cluster / per-image family_idx are present for the ABC loss term during validation
    (with the standard non-grouped val batches, ABC episodes rarely form, so abc_loss stays near 0).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize and force the task to 'pose-prompt'."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "pose-prompt"

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a PosePromptDataset (so cluster/family_idx exist for standalone validation)."""
        cfg = self.args
        return PosePromptDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or (mode == "val"),
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=self.stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="pose-prompt",
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
