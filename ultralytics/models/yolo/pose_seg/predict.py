# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PoseSegPredictor(PosePredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose-segmentation'

    def postprocess(self, preds, img, orig_imgs):
        raise NotImplementedError("Does not process segmentation results for pred yet.")
