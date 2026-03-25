import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Sampler

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first


class FamilyBatchSampler(Sampler):
    """Yields batches that sample from each family subdir (family_N and orphan).

    Each batch is filled by round-robining across all families, so every batch
    contains images from multiple families. Within a family, images are shuffled
    each epoch.

    Supports DDP: all ranks generate identical global batches using a shared seed,
    then each rank yields only its own slice (indices rank*batch_size to (rank+1)*batch_size).
    """

    def __init__(self, im_files: list[str], batch_size: int, rank: int = -1, n_gpus: int = 1, seed: int = 0):
        self.batch_size = batch_size  # per-GPU batch size
        self.rank = rank
        self.n_gpus = n_gpus
        self.seed = seed
        self.epoch = 0

        family_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, path in enumerate(im_files):
            family = Path(path).parent.name  # e.g. "family_0", "family_1", "orphan"
            family_to_indices[family].append(i)

        self.families = sorted(family_to_indices.keys())
        self.family_indices = dict(family_to_indices)
        global_batch_size = batch_size * max(n_gpus, 1)
        self.global_batch_size = global_batch_size  # number of samples per batch across all GPUs
        
        # Compute number of batches per epoch based on the largest family size
        max_family_size = max(len(v) for v in self.family_indices.values())
        self.n_batches = max_family_size * len(self.families) // global_batch_size

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so each epoch gets a different shuffle order."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        
        # Map each family to shuffled image indices for this epoch (e.g., "family_0": [5, 2, 8, ...])
        img_indices_map = {f: rng.sample(idxs, len(idxs)) for f, idxs in self.family_indices.items()}

        n_families = len(self.families)  # number of families in the dataset

        # Decide number of samples per family for each batch
        n_samples_per_family, remainder = divmod(self.global_batch_size, n_families)

        for _ in range(self.n_batches):
            global_batch = []  # a batch containing image indicess for all GPUs
            families_shuffled = rng.sample(self.families, n_families)  # shuffled family order
            for i, family in enumerate(families_shuffled):
                # number of samples to draw from this family for this batch
                n = n_samples_per_family + (1 if i < remainder else 0)

                img_indices_family = img_indices_map[family]  # available image indices in this family
                if len(img_indices_family) < n:
                    # If not enough images in this family, fill the rest by sampling from the same family
                    img_indices_family += rng.choices(self.family_indices[family], k=(n - len(img_indices_family)))
                global_batch.extend(img_indices_family[:n])  # add n samples from this family to the global batch
                img_indices_map[family] = img_indices_family[n:]  # remove the used indices from this iteration
            rng.shuffle(global_batch)

            if self.rank == -1:  # if single GPU
                yield global_batch
            else:
                start = self.rank * self.batch_size
                yield global_batch[start : start + self.batch_size]

    def __len__(self) -> int:
        return self.n_batches


class FamilyPoseTrainer(PoseTrainer):
    """PoseTrainer that replaces the training dataloader with a family-balanced batch sampler."""

    def _model_train(self):
        """Set epoch on the family sampler before each epoch (not just in DDP mode)."""
        super()._model_train()
        if RANK == -1:  # if single GPU                                                                                                                                                                                                             
            self.train_loader.sampler.set_epoch(self.epoch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            # Use standard dataloader for validation
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        n_gpus = torch.distributed.get_world_size() if rank != -1 else 1
        batch_sampler = FamilyBatchSampler(dataset.im_files, batch_size, rank=rank, n_gpus=n_gpus)

        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)

        generator = torch.Generator()
        generator.manual_seed(6148914691236517205)

        loader = InfiniteDataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
        )

        loader.sampler = batch_sampler
        return loader
