# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from copy import copy

import torch

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import PosePromptDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import PosePromptModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .sampler import GroupedFamilySampler


class PosePromptTrainer(yolo.pose.PoseTrainer):
    """Trainer for the "pose-prompt" task: pose + per-anchor embedding + episodic few-shot ABC head.

    Uses PosePromptDataset (per-box cluster + per-image family_idx) and a
    GroupedFamilySampler so each training batch contains same-family image groups (the
    within-family co-occurrence the ABC loss needs). Validation uses the standard dataloader.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the trainer and force the task to 'pose-prompt'.

        Note: we skip PoseTrainer.__init__ and call DetectionTrainer.__init__ directly, because
        PoseTrainer.__init__ unconditionally sets overrides["task"] = "pose" (which would land
        runs under runs/pose and set args.task wrong). All of PoseTrainer's *methods* are still
        inherited via the class hierarchy; only its task-clobbering __init__ is bypassed.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose-prompt"
        yolo.detect.DetectionTrainer.__init__(self, cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a PosePromptModel configured from the dataset."""
        model = PosePromptModel(
            cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            na=self.data["na"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a PosePromptDataset for the given split."""
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
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
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Use the grouped family sampler for training; standard dataloader for validation.

        batch_size here is per-rank (the trainer divides the global batch by world size). The
        family sampler is a per-sample Sampler that shards whole groups across DDP ranks, so it
        plugs into the normal DataLoader/DDP path (sampler= + batch_size + drop_last).
        """
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        k = int(getattr(self.args, "family_group_size", 4))
        sampler = GroupedFamilySampler(
            dataset.im_files,
            batch_size,
            k=k,
            seed=self.args.seed,
            rank=rank,
            num_replicas=max(getattr(self, "world_size", 1), 1),
        )

        nd = os.cpu_count() or 1
        num_workers = min(nd // max(getattr(self, "world_size", 1), 1), self.args.workers)
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # the sampler does the shuffling
            drop_last=True,  # keep every batch a full groups_per_batch x k
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def _model_train(self):
        """Reshuffle the grouped sampler each epoch (the trainer only calls set_epoch itself in DDP)."""
        super()._model_train()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

    def get_validator(self):
        """Return a PosePromptValidator; the ABC loss adds an 'abc_loss' term."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "attr_loss", "abc_loss"
        return yolo.pose_prompt.PosePromptValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
