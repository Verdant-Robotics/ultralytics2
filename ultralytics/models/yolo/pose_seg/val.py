# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from ultralytics.models.yolo.pose import PoseValidator


class PoseSegValidator(PoseValidator):
    """ 
    Validation of Pose-Segmentation model is pretty much same as Pose model as ground truth labels for 
    segmentation objects are not available since training uses ground truth bbox labels for this segmentation.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize a 'PoseSegValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = 'pose-segmentation'

    def init_metrics(self, model):
        super().init_metrics(model)
        self.seg_ch_num = model.seg_ch_num

    def process_seg_result(self, preds):
        """
        Input: 
            preds = x_flat, ([P1, P2, P3], kpt)
            Each Pi is (bs, self.no, h_i, w_i), with h_i and w_i being different for each P. e.g 8x8, 4x4, 2x2 corresponding to resolution(self.stride)
            x_flat = (bs, xyxy(bbox),cls0,..,clsi,seg_obj0,seg_obj1,seg0,...,segj, A) e.g A = anchors_len = 8x8 + 4x4 + 2x2 = 84
        Output:
            seg_obj (B, 1, A)
            seg_cls (B, seg_ch_num, A)
            Pi_list for reconstructing anchors later on
        """
        Pi_list = preds[1][0]
        x_flat = preds[0]
        seg_offset = 4 + self.nc
        seg_obj1 = x_flat[:, seg_offset + 1 : seg_offset + 2, :]
        seg_cls = x_flat[:, seg_offset + 2 : seg_offset + 2 + self.seg_ch_num, :]
        return seg_obj1, seg_cls, Pi_list

    def postprocess(self, preds: torch.Tensor):
        return super().postprocess(preds[0]), self.process_seg_result(preds)

    def update_metrics(self, preds, batch):
        preds = preds[0]
        return super().update_metrics(preds, batch)
    
    def plot_predictions(self, batch, preds, ni):
        return super().plot_predictions(batch, preds[0], ni)
