# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import BoxInstModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import unwrap_model

class BoxInstTrainer(yolo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a BoxInstTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "box-inst"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')


    def _setup_train(self):
        """Resolve the pairwise warmup fraction to an absolute iteration count once dataloaders are built."""
        super()._setup_train()
        total_iters = len(self.train_loader) * self.epochs
        model = unwrap_model(self.model)
        model.pairwise_warmup_iters = max(1, round(model.pairwise_warmup_frac * total_iters))
        LOGGER.info(
            f"BoxInst pairwise warmup: {model.pairwise_warmup_iters}/{total_iters} iters "
            f"({model.pairwise_warmup_frac:.0%} of training)"
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = BoxInstModel(cfg, ch=3, nc=self.data['nc'], na=self.data['na'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of BoxInstModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data['kpt_shape']


    def get_validator(self):
        """Returns an instance of the BoxInstValidator class for validation."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss', 'attr_loss', 'maxlabel_loss', 'pair_loss'
        return yolo.box_inst.BoxInstValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    