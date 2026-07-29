# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG


class PosePromptPredictor(PosePredictor):
    """Predictor for the "pose-prompt" task.

    The standard inference tensor is identical to Pose (box + cls + attributes + keypoints), so pose
    prediction/decoding is inherited unchanged. Per-anchor embeddings and the example-conditioned
    ("ABC") classification are handled separately via PosePromptModel (embeddings ride in the
    model's auxiliary output; classify_embeddings runs the ABC head on cached embeddings).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize and force the task to 'pose-prompt'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose-prompt"
