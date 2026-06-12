# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from ultralytics.models.yolo.pose import PoseValidator


class BoxInstValidator(PoseValidator):

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize a 'BoxInstValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'box-inst'
