# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.na = m.na  # number of attributes
        self.no = m.nc + m.na + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def unflatten_batch(self, objects: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Converts a single list containing all objects for a batch to a tensor whose first dimension
        is the batch index.
        """
        if objects.shape[0] == 0:
            max_objects_per_image = torch.tensor(0, device=self.device)
        else:
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            max_objects_per_image = counts.max()
        out = torch.zeros(batch_size, max_objects_per_image, *objects.shape[1:], device=self.device)
        for i in range(batch_size):
            matches = (batch_idx == i).view(-1)
            out[i, 0:matches.sum()] = objects[matches, :]
        return out

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        batch_idx = targets[:, 0]
        objects = targets[:, 1:]
        out = self.unflatten_batch(objects, batch_idx, batch_size)
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def calculate_attribute_loss(self, batch, pred_attributes, target_gt_idx, fg_mask):
        """
        pred_attributes: (bs, n_anchors, n_attributes)
        target_gt_idx: (bs, n_anchors) For each anchor, index of the assigned ground truth object.
        fg_mask: (bs, n_anchors) Boolean mask indicating which anchors have been matched to a gt object.
        """
        flat_attributes = batch['attributes'].to(self.device).float()  # (n_targets_in_batch, n_attributes)
        batch_idx = batch['batch_idx'].view(-1, 1)
        batch_size = pred_attributes.shape[0]
        dtype = pred_attributes.dtype

        # attributes, attribute_labels, attribute_mask: (bs, max_objects_per_image, n_attributes)
        attributes = self.unflatten_batch(flat_attributes, batch_idx, batch_size)
        attribute_labels = torch.clamp(attributes, 0, 1).to(dtype=dtype)
        attribute_mask = (attributes >= 0)  # -1 means ignore (note that filler targets are 0)
        num_attributes = attribute_labels.shape[2]  # Number of attributes per object
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).expand(-1, -1, num_attributes)  # (bs, n_anchors, n_attributes)
        anchor_attributes = torch.gather(attribute_labels, 1, target_gt_idx_expanded)
        anchor_attribute_mask = torch.minimum(
            torch.gather(attribute_mask, 1, target_gt_idx_expanded),
            fg_mask.unsqueeze(-1).expand(-1, -1, num_attributes),  # Removes unassigned anchors
        )

        attribute_losses = F.binary_cross_entropy_with_logits(
            pred_attributes,
            anchor_attributes,
            reduction="none",
        )
        attribute_losses *= anchor_attribute_mask
        num_attribute_labels = max(anchor_attribute_mask.sum(), 1)

        # fg_mask has shape (bs, n_anchors)
        return attribute_losses[anchor_attribute_mask].sum() / num_attribute_labels

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, attributes
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, pred_attributes = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.na), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_attributes = pred_attributes.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # target_scores: (b, n_anchors, nc) 1-hot class label or 0 if the anchor is not matched
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        if 'attributes' in batch:
            loss[3] = self.calculate_attribute_loss(batch, pred_attributes, target_gt_idx, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.attr  # attribute gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(6, device=self.device)  # box, cls, attributes, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        batch_size = feats[0].shape[0]
        pred_distri, pred_scores, pred_attributes = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in feats], 2
        ).split(
            (self.reg_max * 4, self.nc, self.na), 1
        )

        loss = self.calculate_bbox_kpt_loss(
            loss, batch, feats, pred_distri, pred_scores, pred_attributes, pred_kpts
        )

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def calculate_bbox_kpt_loss(
        self, loss, batch, feats, pred_distri, pred_scores, pred_attributes, pred_kpts
    ):
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_attributes = pred_attributes.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # (B, T, 1), (B, T, 4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, h x w, 4(xyxy))
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts, batch['ignore_kpt'])

            if 'attributes' in batch:
                loss[5] = self.calculate_attribute_loss(batch, pred_attributes, target_gt_idx, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.attr  # attribute gain
        return loss

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
        ignore_kpt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        for i in range(batch_size):
            if ignore_kpt[i]:
                # We should ignore the keypoints for this image
                # We replace gt keypoints with pred keypoints so distance is 0. kpts_loss will be 0
                selected_keypoints[i] = pred_kpts[i]

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss
    

class PosePromptLoss(v8PoseLoss):
    """v8PoseLoss extended with an episodic, example-conditioned few-shot ("ABC") loss.

    Adds a 7th loss term (order: box, pose, kobj, cls, dfl, attr, abc). The ABC term is built from
    within-batch, within-family episodes (see calculate_abc_loss):
      * within each family the labelled clusters are randomly partitioned into up to abc_num_slots
        classes (each a set of one or more clusters), padded with an "empty" sentinel;
      * each class prototype is a random positive mixture of its clusters' embeddings;
      * every foreground anchor is classified by the head's abc_head against those class prototypes
        with a partial-label (allowed-set) masked-softmax loss that exploits the cluster sign
        convention (0 = unknown, >0 = known non-exhaustive, <0 = known exhaustive).

    No hooks or assigner wrapping: the embeddings are a first-class head output (pred_embed) and
    the assignment (fg_mask / target_gt_idx) is a local computed here, exactly as
    calculate_attribute_loss already consumes it.
    """

    def __init__(self, model):
        """Initialize PosePromptLoss and cache a reference to the head's ABC module."""
        super().__init__(model)
        self.abc_head = model.model[-1].abc_head
        self.abc_gain = getattr(self.hyp, "abc", 0.5)
        # Fixed number of prototype slots per episode; unused slots are padded with the empty sentinel
        # so the net learns its "absent" meaning. Should match export num_protos (deploy max classes).
        self.abc_num_slots = getattr(self.hyp, "abc_num_slots", 3)
        # Concentration of the per-cluster stick-breaking mixture (see calculate_abc_loss). Smaller =
        # prototypes built from fewer members regardless of how many are present; a=1 => Beta(1,1)=Uniform.
        self.abc_stick_alpha = getattr(self.hyp, "abc_stick_alpha", 1.0)

    def __call__(self, preds, batch):
        """Compute the total loss (box, pose, kobj, cls, dfl, attr, abc)."""
        loss = torch.zeros(7, device=self.device)  # box, pose, kobj, cls, dfl, attr, abc
        feats, pred_kpts, pred_embed = preds if isinstance(preds[0], list) else preds[1]
        batch_size = feats[0].shape[0]
        pred_distri, pred_scores, pred_attributes = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc, self.na), 1)

        loss = self.calculate_bbox_kpt_loss(
            loss, batch, feats, pred_distri, pred_scores, pred_attributes, pred_kpts, pred_embed
        )
        return loss.sum() * batch_size, loss.detach()

    def calculate_bbox_kpt_loss(
        self, loss, batch, feats, pred_distri, pred_scores, pred_attributes, pred_kpts, pred_embed
    ):
        """Standard pose losses (mirrors v8PoseLoss) plus the ABC loss, sharing one assignment."""
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_attributes = pred_attributes.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        pred_embed = pred_embed.permute(0, 2, 1).contiguous()  # (B, A, E)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts_dec = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts_dec,
                batch["ignore_kpt"],
            )
            if "attributes" in batch:
                loss[5] = self.calculate_attribute_loss(batch, pred_attributes, target_gt_idx, fg_mask)
            if "cluster" in batch and "family_idx" in batch:
                loss[6] = self.calculate_abc_loss(batch, pred_embed, fg_mask, target_gt_idx)

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl
        loss[5] *= self.hyp.attr
        loss[6] *= self.abc_gain
        return loss

    def calculate_abc_loss(self, batch, pred_embed, fg_mask, target_gt_idx):
        """Episodic partial-label few-shot loss.

        Per family: gather all foreground anchor embeddings and their clusters, then partition the
        available clusters into a RANDOM number (1..min(n_clusters, abc_num_slots)) of non-overlapping
        classes - each class is a SET of one or more clusters (multi-cluster classes), and some clusters
        are left unassigned so their queries become NOTA. Each cluster gets a random positive-mixture
        prototype (Exponential(1) weight per member), L2-normalized so cluster size does not matter and
        scaled by a random weight in [1, 10]; a class prototype is the sum of its clusters' prototypes
        (so several clusters can share a class, each contributing a random amount). The prototype set is
        padded to abc_num_slots slots with the "empty" sentinel (1, 0, ..., 0). Every foreground
        anchor is then classified against those slots with a partial-label (allowed-set) masked-softmax
        loss; sentinel (padding) slots are never an allowed label, so the net learns the sentinel means
        "class absent". Randomising the class count and cluster-to-class assignment directly trains the
        arbitrary-subset behaviour used at inference. No support/query split: prototypes and queries
        share the same pool; the random mixture keeps a prototype from collapsing onto a single query.

        Args:
            batch (dict): must contain "cluster" (N,) per-box cluster ids and "family_idx" (B,).
            pred_embed (torch.Tensor): (B, A, E) per-anchor embeddings.
            fg_mask (torch.Tensor): (B, A) bool, foreground (matched) anchors.
            target_gt_idx (torch.Tensor): (B, A) index of the assigned GT per anchor.

        Returns:
            (torch.Tensor): scalar ABC loss (0 if no family has a labelled cluster).
        """
        device = self.device
        B = fg_mask.shape[0]
        k_max = self.abc_num_slots
        # Compute the episode in the ABC head's own parameter dtype (fp32 in training; fp16 when the
        # validator runs a half() model under AMP). Matching it avoids fp32/fp16 clashes in index_add_
        # and the head's matmuls; the head call below also runs with autocast off so nothing re-casts.
        abc_dtype = next(self.abc_head.parameters()).dtype

        # cluster per GT box -> per anchor (same GT ordering the assigner used)
        flat_cluster = batch["cluster"].to(device).view(-1, 1).float()
        gt_clusters = self.unflatten_batch(flat_cluster, batch["batch_idx"].view(-1, 1), B)[..., 0].long()  # (B, maxb)
        cluster_per_anchor = torch.gather(gt_clusters, 1, target_gt_idx)  # (B, A)

        family_idx = batch["family_idx"]
        family_idx = family_idx.to(device) if torch.is_tensor(family_idx) else torch.as_tensor(family_idx, device=device)

        total_loss = torch.zeros((), device=device)
        n_terms = 0

        for fam in torch.unique(family_idx):
            m = (family_idx == fam)[:, None] & fg_mask  # (B, A) foreground anchors in this family
            emb = pred_embed[m].to(abc_dtype)  # (M, E), matches the ABC head's parameter dtype
            clu = cluster_per_anchor[m]        # (M,)

            # Map the signed cluster ids to compact ids in one pass: torch.unique's inverse IS the
            # value->id map already applied to every anchor (no per-anchor binary search). `cid` indexes
            # everything below directly. The unknown cluster (id 0) is just another id here - it gets a
            # prototype too, but is simply never assigned to a class.
            uniq, cid = torch.unique(clu, return_inverse=True)  # uniq sorted; cid (M,) in 0..len(uniq)-1
            real = uniq != 0  # (n_ids,) which cluster ids are real examples (not the unknown cluster)
            n_clusters = int(real.sum())
            if n_clusters == 0:
                continue
            cluster_is_exhaustive = uniq < 0  # (n_ids,) whether each cluster was exhaustively labelled (< 0)

            # Per-cluster random-mixture prototypes (one per cluster, unknown included): combine each
            # cluster's members with random weights, L2-normalize (so cluster size does not affect the
            # direction), then scale by a random weight in [1, 10] - modelling a user giving a few-to-many
            # examples of each cluster when several clusters share a class.
            # The mixture weights come from truncated Dirichlet-process stick-breaking with concentration
            # a = abc_stick_alpha (i.e. Dirichlet(a/n, ..., a/n)). Unlike a flat Dirichlet(1, ..., 1) -
            # whose weights collapse to the near-uniform 1/n mean as the member count n grows (a noisy
            # average) - this keeps the mass on O(a) members regardless of n, so a prototype can still
            # look like a single (or few) example(s) when many are present. b_i ~ Beta(1, a),
            # pi_i = b_i * prod_{j<i}(1 - b_j), with the last (random) member forced to b = 1 so the
            # leftover stick mass is assigned to it (a=1 => Beta(1,1) => Uniform).
            n_uniq = uniq.numel()
            n_mem = cid.shape[0]
            counts = torch.bincount(cid, minlength=n_uniq)  # members per cluster
            max_n = int(counts.max())
            # Lay members out in a (n_uniq, max_n) grid in random order within each cluster (sort by
            # cluster id + noise), so the decreasing stick weights land on members at random.
            sort_idx = torch.argsort(cid.float() + torch.rand(n_mem, device=device))
            scid = cid[sort_idx]
            ramp = torch.arange(n_mem, device=device)
            boundary = torch.cat([torch.ones(1, dtype=torch.bool, device=device), scid[1:] != scid[:-1]])
            col = ramp - torch.where(boundary, ramp, torch.zeros_like(ramp)).cummax(0).values  # position in cluster
            cols = torch.arange(max_n, device=device)
            valid = cols[None, :] < counts[:, None]
            last = cols[None, :] == (counts[:, None] - 1)
            # Stick weights in the embedding dtype: far-tail members may underflow to 0 under fp16, which
            # is harmless (their weight is negligible), and the small clusters that matter for aggregation
            # (n = 2, 3) never underflow. The prototype is L2-normalized below, so lost tail mass is moot.
            u = torch.rand(n_uniq, max_n, device=device, dtype=emb.dtype)
            b = 1.0 - (1.0 - u) ** (1.0 / self.abc_stick_alpha)  # Beta(1, a) via inverse CDF
            b = torch.where(last, torch.ones_like(b), b)
            b = torch.where(valid, b, torch.zeros_like(b))  # invalid slots: b=0 => (1-b)=1 => pi=0
            prev = torch.cat(
                [torch.ones(n_uniq, 1, device=device, dtype=emb.dtype), torch.cumprod(1.0 - b, dim=1)[:, :-1]],
                dim=1,
            )  # prod_{j<i}(1 - b_j)
            pi = b * prev  # (n_uniq, max_n); each cluster's weights sum to 1
            emb_grid = torch.zeros(n_uniq, max_n, emb.shape[1], device=device, dtype=emb.dtype)
            emb_grid[scid, col] = emb[sort_idx]  # differentiable scatter: grad flows back to members
            cproto = (pi[:, :, None] * emb_grid).sum(dim=1)  # (n_uniq, E) per-cluster weighted mixture
            cproto = F.normalize(cproto, dim=-1) * torch.empty(
                n_uniq, 1, device=device, dtype=emb.dtype
            ).uniform_(1.0, 10.0)

            # Partition the real clusters into n_real non-overlapping classes (each a set of one or more
            # clusters); some clusters stay unassigned so their queries become NOTA. cls_of_cluster[c] is
            # the class of cluster id c (-1 = unassigned; the unknown cluster is never assigned).
            # Draw the class count twice and take the max: this biases toward MORE classes, so when a
            # family has several clusters they are more often split into separate classes (kept as
            # distinct examples) rather than merged into one class or dropped to NOTA - which is where
            # the weed-vs-weed separation signal comes from.
            n_real = int(torch.randint(1, min(n_clusters, k_max) + 1, (2,), device=device).max())
            order = real.nonzero().flatten()[torch.randperm(n_clusters, device=device)]  # shuffled real ids
            cls_of_cluster = torch.full((uniq.numel(),), -1, dtype=torch.long, device=device)
            cls_of_cluster[order[:n_real]] = torch.arange(n_real, device=device)  # seed one cluster per class
            if n_clusters > n_real:
                rest = torch.randint(0, n_real + 1, (n_clusters - n_real,), device=device)  # n_real => unassigned
                rest[rest == n_real] = -1
                cls_of_cluster[order[n_real:]] = rest

            # Each class prototype is the sum of its clusters' prototypes.
            assigned = cls_of_cluster >= 0
            real_protos = torch.zeros(n_real, emb.shape[1], device=device, dtype=emb.dtype)
            real_protos.index_add_(0, cls_of_cluster[assigned], cproto[assigned])

            # Pad the prototype set to k_max slots with the "empty" sentinel (1, 0, ..., 0).
            pad = torch.zeros(k_max - n_real, emb.shape[1], device=device, dtype=emb.dtype)
            pad[:, 0] = 1.0
            protos = torch.cat([real_protos, pad], 0)  # (k_max, E)

            # Run the head with autocast off: emb/protos and the head's params are all fp32, so this
            # keeps the transformer/matmuls consistently fp32 (autocast would otherwise mix fp16/fp32).
            with torch.autocast(device_type=self.device.type, enabled=False):
                logits = self.abc_head(emb, protos)  # (M, k_max+1)   every fg anchor is a query
            # Per-anchor class (unknown/unassigned clusters map to -1). A class is exhaustive iff ALL its
            # clusters are; padding slots are marked exhaustive too, so they are never an allowed label
            # and _abc_class_labels already returns the full k_max width.
            cls_of_query = cls_of_cluster[cid]
            class_is_exhaustive = torch.ones(k_max, dtype=torch.bool, device=device)
            for k in range(n_real):
                class_is_exhaustive[k] = cluster_is_exhaustive[cls_of_cluster == k].all()
            labels = self._abc_class_labels(cls_of_query, clu == 0, class_is_exhaustive)  # (M, k_max+1)
            # A query is "trivial" when every class is an allowed label (nothing excluded, so num == den);
            # drop those from the mean. Not all are trivial: the >=1 seeded cluster gives a matched query.
            trivial = labels.all(dim=1)

            neg_inf = torch.finfo(logits.dtype).min
            num = torch.logsumexp(logits.masked_fill(~labels, neg_inf), dim=1)
            den = torch.logsumexp(logits, dim=1)
            per_query = den - num  # -log(sum_{S} softmax) >= 0
            total_loss = total_loss + per_query[~trivial].mean()
            n_terms += 1

        return total_loss / n_terms if n_terms else torch.zeros((), device=device)

    @staticmethod
    def _abc_class_labels(cls_of_query, is_unknown, class_is_exhaustive):
        """Build the (M, K+1) boolean class-label mask (K class slots + NOTA).

        Inputs (M = number of query anchors, K = number of class slots):
          cls_of_query (M,):        class index of each query's cluster (-1 = unassigned / not an example)
          is_unknown (M,):          queries whose cluster is unknown (cluster 0)
          class_is_exhaustive (K,): classes whose clusters are ALL exhaustively labelled
        Allowed labels per query:
          known, in class k         -> {k}
          known, in no class        -> {NOTA}
          unknown (cluster 0)        -> {NOTA} U {classes that are NOT fully exhaustive}
        (An unknown plant cannot be an exhaustively-labelled cluster, so fully-exhaustive classes are
        excluded for it.)
        """
        M, n_cls = cls_of_query.shape[0], class_is_exhaustive.shape[0]
        nota = n_cls
        labels = torch.zeros(M, n_cls + 1, dtype=torch.bool, device=cls_of_query.device)
        matched = cls_of_query >= 0  # in a class (unknown queries have cls_of_query == -1, so this implies known)
        labels[matched, cls_of_query[matched]] = True  # in a class -> that class
        labels[~matched, nota] = True  # not in a class (known-but-unassigned, or unknown) -> NOTA
        labels[:, :n_cls] |= is_unknown[:, None] & ~class_is_exhaustive[None, :]  # unknown -> also non-exhaustive classes
        return labels


class v8PoseSegLoss(v8PoseLoss):
    class PredE(Enum):
        FEAT = 0
        DIST = 1
        SCORES = 2
        ATTR = 3
        KPTS = 4
        OBJ0_CLS = 5
        OBJ1_CLS = 6

    def __init__(self, model):
        super().__init__(model)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.seg_ch_num = model.model[-1].seg_ch_num
        self.no = model.model[-1].no

    def prepare_preds(self, preds):
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        all_preds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores, pred_attributes, pred_obj0_cls, pred_obj1_cls = all_preds.split((self.reg_max * 4, self.nc, self.na, self.seg_ch_num, self.seg_ch_num), 1)  # B, S, A
        return {
            self.PredE.FEAT: feats,
            self.PredE.DIST: pred_distri,
            self.PredE.SCORES: pred_scores,
            self.PredE.ATTR: pred_attributes,
            self.PredE.KPTS: pred_kpts,
            self.PredE.OBJ0_CLS: pred_obj0_cls,
            self.PredE.OBJ1_CLS: pred_obj1_cls
        }

    def __call__(self, preds, batch):
        return NotImplementedError


class v8PSLPose(v8PoseSegLoss):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, preds, batch):
        loss = torch.zeros(6, device=self.device)  # box, cls, attributes, dfl, kpt_location, kpt_visibility
        preds_dict = self.prepare_preds(preds)
        feats, pred_distri, pred_scores, pred_attributes, pred_kpts = preds_dict[self.PredE.FEAT], preds_dict[self.PredE.DIST], preds_dict[self.PredE.SCORES], preds_dict[self.PredE.ATTR], preds_dict[self.PredE.KPTS]
        batch_size = pred_distri.shape[0]
        loss = self.calculate_bbox_kpt_loss(loss=loss, batch=batch, feats=feats, pred_distri=pred_distri, pred_scores=pred_scores, pred_attributes=pred_attributes, pred_kpts=pred_kpts)
        return loss.sum() * batch_size, loss.detach()

    
class PoseLossBoxInst(v8PoseLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.seg_ch_num = model.model[-1].seg_ch_num
        self.no = model.model[-1].no

    def __call__(self, preds, batch):
        loss = torch.zeros(6, device=self.device)  # box, cls, attributes, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        all_preds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores, pred_attributes, _ = all_preds.split((self.reg_max * 4, self.nc, self.na, self.seg_ch_num), 1)  # B, S, A        
        loss = self.calculate_bbox_kpt_loss(loss=loss, batch=batch, feats=feats, pred_distri=pred_distri, pred_scores=pred_scores, pred_attributes=pred_attributes, pred_kpts=pred_kpts)
        batch_size = pred_distri.shape[0]
        return loss.sum() * batch_size, loss.detach()


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
