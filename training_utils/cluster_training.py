from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils import colorstr
from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.torch_utils import unwrap_model

from family_sampler import FamilyPoseTrainer


class EmbeddingHead(nn.Module):
    """Embedding branch per FPN scale (e.g. P3, P4, P5)
    """

    def __init__(self, in_channels: list[int], embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.cv = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, embed_dim, 1, bias=False),
            )
            for c in in_channels
        ])

    def forward(self, fpn_features: list[torch.Tensor]) -> torch.Tensor:
        feats = []
        for cv, f in zip(self.cv, fpn_features):
            B, _, H, W = f.shape
            out = cv(f)  # (B, embed_dim, H, W)
            feats.append(out.reshape(B, self.embed_dim, H * W))
        return torch.cat(feats, dim=2)  # (B, embed_dim, n_grid_points)


class _FeatureHook:
    """Picklable forward pre-hook that runs EmbeddingHead before the detect head and caches embeddings."""

    def __init__(self, cached_features: dict, embedding_head: EmbeddingHead):
        self.cached_features = cached_features
        self.embedding_head = embedding_head

    def __call__(self, _, args):
        # args[0] is FPN feature maps in different scales (e.g. P3, P4, P5) as inputs to the Detect head
        self.cached_features["features"] = self.embedding_head(args[0])  # (B, embed_dim, n_grid_points)


class CachingAssigner:
    """Wraps TaskAlignedAssigner to cache fg_mask and target_gt_idx after each call."""

    def __init__(self, assigner, cache):
        self._assigner = assigner
        self._cache = cache

    def __call__(self, *args, **kwargs):
        result = self._assigner(*args, **kwargs
                                )  # target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
        self._cache["fg_mask"] = result[3]
        self._cache["target_gt_idx"] = result[4]
        return result

    def __getattr__(self, name):
        if name == "_assigner":
            raise AttributeError(name)
        return getattr(self._assigner, name)
    

class ClusterYOLODataset(YOLODataset):
    """YOLODataset subclass that additionally loads the cluster column from label files."""

    def get_labels(self):
        labels = super().get_labels()
        for i, label_file in enumerate(img2label_paths(self.im_files)):
            n = len(labels[i]["cls"])
            if not Path(label_file).exists() or n == 0:
                labels[i]["cluster"] = np.full(0, -1, dtype=np.int64)
                continue
            labels[i]["cluster"] = np.loadtxt(label_file, dtype=np.float32, ndmin=2)[:, -1].astype(np.int64)
        return labels

    def update_labels(self, include_class):
        """Filter cluster arrays to remove boxes whose class is not in include_class.
        Cluster array is a 1D array of length n_boxes, where each element is the cluster id of the corresponding GT box.

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
        """
        # Apply box filter mask to cluster
        if include_class is not None:
            include_class_array = np.array(include_class).reshape(1, -1)
            for i, label in enumerate(self.labels):
                if "cluster" in label:
                    # Get a boolean mask of boxes to keep only boxes whose class is in include_class
                    j = (label["cls"] == include_class_array).any(1)  # (n_boxes,)
                    # Apply the mask to cluster array to keep clusters of the boxes whose class is in include_class
                    self.labels[i]["cluster"] = label["cluster"][j]  # (n_boxes_filtered,)
        super().update_labels(include_class)

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        cluster = self.labels[index].get("cluster")
        if cluster is not None:
            # Append cluster to apply the same box-removal filtering to cluster during augmentations
            label["cls"] = np.concatenate(
                [label["cls"], cluster.astype(np.float32).reshape(-1, 1)], axis=1
            )  # becomes [class_id | attr0 ... attr_na | cluster]
        return label

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        attrs = sample["attributes"]  # (n_boxes, na+1)
        # Separate cluster from attributes
        sample["cluster"] = attrs[:, -1].long()  # (n_boxes,)
        sample["attributes"] = attrs[:, :-1]  # restore to (n_boxes, na)
        return sample

    @staticmethod
    def collate_fn(batch):
        clusters = [b.pop("cluster") for b in batch]
        collated = YOLODataset.collate_fn(batch)
        collated["cluster"] = torch.cat(clusters, dim=0)
        return collated


class ClusterPoseLoss(v8PoseLoss):
    """v8PoseLoss extended with a cluster loss."""

    def __init__(self, model, cached_features):
        super().__init__(model)
        self._model = model
        self.cached_features = cached_features
        self._assignment_cache = {}
        self.assigner = CachingAssigner(self.assigner, self._assignment_cache)
        self.cluster_temp = 0.07

    def __call__(self, preds, batch):
        total_loss, loss_items = super().__call__(preds, batch)
        cluster_loss_item = torch.zeros(1, device=self.device)
        # if self._model.training and "cluster" in batch and "features" in self.cached_features:
        if self._model.training:
            assert "cluster" in batch, "Cluster labels not found in batch."
            assert "features" in self.cached_features, "FeatureHook was not registered."
            batch_size = batch["img"].shape[0]
            cluster_loss = self.calculate_cluster_loss(batch)
            cluster_weight = getattr(self.hyp, "cluster", 1.0)
            total_loss = total_loss + (cluster_loss * cluster_weight) * batch_size
            cluster_loss_item = cluster_loss.detach().unsqueeze(0)
        loss_items = torch.cat([loss_items, cluster_loss_item])
        return total_loss, loss_items

    def _unflatten_cluster(self, flat_cluster, batch_idx, batch_size):
        """Unflatten cluster labels with -1 padding (0 is a valid cluster id so can't use zeros)."""
        batch_idx = batch_idx.view(-1)
        if flat_cluster.shape[0] == 0:
            return torch.full((batch_size, 0), -1, dtype=torch.long, device=self.device)
        _, counts = batch_idx.unique(return_counts=True)
        max_boxes = counts.max().item()
        out = torch.full((batch_size, max_boxes), -1, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            mask = batch_idx == i
            n = mask.sum()
            out[i, :n] = flat_cluster[mask]
        return out

    def calculate_cluster_loss(self, batch):
        """
        Cluster loss: for each family in the batch:
            1. Collect foreground grid points with cluster >= 0
            2. Skip if fewer than 2 distinct clusters
            3. Group feature vectors of foreground points by cluster label
            4. Compute prototype (mean feature vector) per group
            5. Apply cross-entropy loss between feature vectors @ prototypes and cluster labels
        """
        fg_mask = self._assignment_cache["fg_mask"]              # (B, n_grid_points)
        target_gt_idx = self._assignment_cache["target_gt_idx"]  # (B, n_grid_points)
        features = self.cached_features["features"].permute(0, 2, 1)  # (B, n_grid_points, C)
        B = fg_mask.shape[0]
        flat_cluster = batch["cluster"].to(self.device)          # (N_total,)

        # Maps boxes to clusters
        gt_clusters_padded = self._unflatten_cluster(
            flat_cluster, batch["batch_idx"], B
        )                                                        # (B, n_max_boxes)
        
        # Maps grid points to clusters (points to boxes to grid points)
        cluster_per_anchor = torch.gather(gt_clusters_padded, 1, target_gt_idx)  # (B, n_grid_points)

        cluster_per_fg = cluster_per_anchor[fg_mask]             # (n_fg,)
        features_per_fg = features[fg_mask]                      # (n_fg, C)
        family_per_fg = batch["family_idx"].unsqueeze(1).expand_as(fg_mask)[fg_mask]  # (n_fg,)

        cluster_loss = torch.tensor(0.0, device=self.device)
        n_families = 0

        for family_id in batch["family_idx"].unique():
            fam = family_per_fg == family_id                     # (n_fg,)
            clusters = cluster_per_fg[fam]                       # (n_fg_fam,)
            feats = features_per_fg[fam]                         # (n_fg_fam, C)

            valid = clusters >= 0                                # exclude non-ClusterSet boxes
            if valid.sum() < 2:
                continue
            clusters = clusters[valid]
            feats = feats[valid]

            unique_clusters = clusters.unique()                  # unique cluster ids within this family
            if len(unique_clusters) < 2:
                continue

            prototypes = torch.stack(
                [feats[clusters == c].mean(0) for c in unique_clusters]
            )                                                    # (n_clusters, C)
            cos_sim = F.normalize(feats, dim=-1) @ F.normalize(prototypes, dim=-1).T  # (n_fg_fam, n_clusters)

            cluster_to_idx = {c.item(): i for i, c in enumerate(unique_clusters)}
            targets = torch.tensor(
                [cluster_to_idx[c.item()] for c in clusters], device=self.device
            )                                                    # (n_fg_fam,)

            cluster_loss = cluster_loss + F.cross_entropy(cos_sim / self.cluster_temp, targets)
            n_families += 1

        cluster_loss = cluster_loss / max(n_families, 1)

        return cluster_loss


class ClusterModelExportWrapper(nn.Module):
    """Wraps the base model for ONNX export, including EmbeddingHead as a second output.

    The standard ONNX export only traces model.forward(), which excludes EmbeddingHead
    since it runs via a forward_pre_hook. This wrapper calls EmbeddingHead explicitly
    inside forward() so the ONNX tracer captures its ops in the graph.

    Outputs:
      - output0: pose detection output (B, n_grid_points, cls + bbox attributes + n_attributes + nkpt*ndim)
      - embeddings: feature vectors at each grid point (B, embed_dim, n_grid_points)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self._base = base_model
        self._embedding_head = base_model.embedding_head
        detect = base_model.model[-1]
        fpn_layer_indices = detect.f
        self._fpn_feats = [None] * detect.nl
        for i, layer_idx in enumerate(fpn_layer_indices):
            base_model.model[layer_idx].register_forward_hook(self._make_fpn_hook(i))

    def _make_fpn_hook(self, i: int):
        def hook(module, args, output):
            self._fpn_feats[i] = output  # capture layer output before detect head processes it
        return hook

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        det = self._base(x)  # detection output, FPN hooks fire → _fpn_feats set
        emb = self._embedding_head(self._fpn_feats)  # (B, embed_dim, n_grid_points)
        return det, emb


class ClusterFamilyPoseTrainer(FamilyPoseTrainer):
    """FamilyPoseTrainer extended with ClusterYOLODataset and ClusterPoseLoss."""

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        cfg = self.args
        return ClusterYOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or (mode == "val"),
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        self._cached_features = {}
        self._register_feature_hook(model)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()  # sets model.args, model.nc, model.kpt_shape, etc.
        unwrap_model(self.model).criterion = ClusterPoseLoss(
            unwrap_model(self.model), self._cached_features
        )

    def _register_feature_hook(self, model):
        m = unwrap_model(model)
        detect = m.model[-1]
        # Use the smallest FPN channel count across scales as embed_dim
        in_channels = [detect.cv2[i][0].conv.in_channels for i in range(detect.nl)]
        embed_dim = min(in_channels)

        m.embedding_head = EmbeddingHead(in_channels, embed_dim)  # attach to model so optimizer includes it
        detect.register_forward_pre_hook(
            _FeatureHook(self._cached_features, m.embedding_head)
        )  # pre-hook will be triggered before the execution of detect head

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (*self.loss_names, "cluster_loss")
        return validator
