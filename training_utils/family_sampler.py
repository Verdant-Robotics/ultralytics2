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
    """Yields batches where each batch contains N images from M randomly sampled families.

    Each batch is structured as M randomly sampled family blocks of N_global images each:
        e.g. [fam0 x N_global | fam1 x N_global | ... | famM-1 x N_global]
    Each GPU receives N sliced images from each family block (rank-slice from shuffled pool)
        e.g., fam0: [img0 ... imgN-1, imgN ... img2N-1, ...]
                          GPU0             GPU1         ...

    N_global: number of images per family per batch across all GPUs
    N: number of images per family per batch per GPU
    M: number of families per batch 

    Images per family are shuffled each epoch.
    Families are drawn from a shuffle to have every family appears at least once per epoch.
    Families that exhaust their images from the shuffled pool are upsampled.
    """

    def __init__(self, im_files: list[str], batch_size: int, rank: int = -1, n_gpus: int = 1, seed: int = 0):
        self.rank = rank
        self.n_gpus = n_gpus
        self.seed = seed
        self.epoch = 0

        family_to_indices: dict[str, list[int]] = defaultdict(list)  # map family to list of image indices
        for i, path in enumerate(im_files):
            family = Path(path).parent.name  # e.g. "family_0", "family_1", or partition name for non-family tiles
            family_to_indices[family].append(i)

        self.families = sorted(family_to_indices.keys())
        self.family_indices = dict(family_to_indices)
        self.global_batch_size = batch_size  # total batch size across all GPUs

        num_images = sum(len(v) for v in self.family_indices.values())
        self.n_batches = num_images // self.global_batch_size  # number of batches per epoch

        n_families = len(self.families)
        per_gpu_batch_size = batch_size // max(n_gpus, 1)
        self.N = max(per_gpu_batch_size // n_families, 3)  # number of images per family per batch per GPU
        self.N_global = self.N * max(n_gpus, 1)            # number of images per family per batch across all GPUs
        self.M = min(n_families, per_gpu_batch_size // self.N)  # number of families per batch

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch so each epoch gets a different shuffle order."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Map each family to shuffled image indices for this epoch
        img_indices_map = {f: rng.sample(indices, len(indices)) for f, indices in self.family_indices.items()}

        # Build a family queue that guarantees every family appears at least once per epoch
        shuffled_families = rng.sample(self.families, len(self.families))
        num_families_epoch = self.n_batches * self.M  # total number of families needed per epoch                                                                                                                                                               
        family_queue = [shuffled_families[i % len(self.families)] for i in range(num_families_epoch)]  

        for i in range(self.n_batches):
            families_batch = family_queue[i * self.M : (i + 1) * self.M]

            # Build global batch
            global_batch = []
            for family in families_batch:
                img_indices_family = img_indices_map[family]
                if len(img_indices_family) < self.N_global:
                    # Upsample if not enough images for this family
                    img_indices_family += rng.choices(self.family_indices[family], k=(self.N_global - len(img_indices_family)))
                global_batch.extend(img_indices_family[:self.N_global])
                img_indices_map[family] = img_indices_family[self.N_global:]

            if self.rank == -1:  # single GPU
                yield global_batch
            else:
                # Each GPU takes its N-sized slice from each family block
                per_gpu = []
                for fam_idx in range(self.M):
                    start = fam_idx * self.N_global + self.rank * self.N
                    per_gpu.extend(global_batch[start : start + self.N])
                yield per_gpu

    def __len__(self) -> int:
        return self.n_batches


class FamilyDataset:
    """Wraps a dataset to add family index to each sample.

    Family index is an integer identifying which family an image belongs to.
    A tensor of family indices of images can be retrieved by "family_idx" key.
    e.g. batch["family_idx"] = tensor([family_idx_img0, family_idx_img1, ...])
    """

    def __init__(self, dataset, family_indices: dict[str, list[int]]):
        self._dataset = dataset
        self._base_collate_fn = getattr(dataset, "collate_fn", None)

        # Map image index to family
        self._img_to_family_map: dict[int, int] = {}
        families = sorted(family_indices.keys())
        for i, family in enumerate(families):
            for idx in family_indices[family]:
                self._img_to_family_map[idx] = i

    def __getitem__(self, index: int) -> dict:
        sample = self._dataset[index]
        sample["family_idx"] = torch.tensor(self._img_to_family_map[index], dtype=torch.long)
        return sample

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate samples, stacking family_idx as a tensor alongside standard fields."""
        family_indices = [b.pop("family_idx") for b in batch]
        collated = self._base_collate_fn(batch)
        collated["family_idx"] = torch.stack(family_indices)
        return collated


class FamilyPoseTrainer(PoseTrainer):
    """PoseTrainer that replaces the training dataloader with a family-balanced batch sampler."""

    def _model_train(self):
        """Set epoch on the family sampler before each epoch (not just in DDP mode)."""
        super()._model_train()
        self.train_loader.batch_sampler.sampler.set_epoch(self.epoch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            # Use standard dataloader for validation
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        n_gpus = self.world_size if rank != -1 else 1
        total_batch_size = batch_size * n_gpus
        batch_sampler = FamilyBatchSampler(dataset.im_files, total_batch_size, rank=rank, n_gpus=n_gpus, seed=self.args.seed)
        dataset = FamilyDataset(dataset, batch_sampler.family_indices)

        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=nd > 0,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )
