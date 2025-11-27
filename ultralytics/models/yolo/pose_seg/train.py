# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseSegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER

class PoseSegTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model='yolo11n-pose.pt', data='coco8-pose.yaml', epochs=3)
        trainer = PoseSegTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseSegTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose-segmentation"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')


    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = PoseSegModel(cfg, ch=3, nc=self.data['nc'], na=self.data['na'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseSegModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data['kpt_shape']


    def get_validator(self):
        """Returns an instance of the PoseSegValidator class for validation."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss', 'attr_loss', 'seg_obj0', 'seg_obj1'
        return yolo.pose_seg.PoseSegValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    