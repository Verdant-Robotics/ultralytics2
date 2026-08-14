# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-sample sampler that orders images into family groups for the pose-prompt task."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Sampler


def _ffd_bin_count(sizes: list[int], k: int) -> int:
    """First-fit-decreasing bin count: minimal-ish number of size-k bins holding all sizes.

    Each size is placed whole (never split) into the first bin with room, opening a new bin otherwise.
    Depends only on the multiset of sizes, so it is shuffle-independent (used to size __len__).
    """
    caps: list[int] = []  # remaining capacity per bin
    for s in sorted(sizes, reverse=True):
        for j in range(len(caps)):
            if caps[j] >= s:
                caps[j] -= s
                break
        else:
            caps.append(k - s)
    return len(caps)


class GroupedFamilySampler(Sampler):
    """Per-sample sampler ordering dataset indices into family groups of size k for the ABC loss.

    It yields a **flat** stream of indices arranged as consecutive k-sized family groups. The
    DataLoader (batch_size = groups_per_batch * k, drop_last=True) then turns every
    groups_per_batch consecutive groups into a batch. Being a per-sample Sampler (not a
    batch_sampler) is what lets it plug into the standard trainer/DDP path: it exposes
    set_epoch (so train_loader.sampler.set_epoch(...) works) and shards whole groups across
    DDP ranks so each rank sees a disjoint subset of the data.

    A "family" is the image's immediate parent directory (family_<N> for cluster tiles; the
    partition dir for familyless tiles). The ABC loss builds one episode per family present in a
    batch, keyed by per-image family_idx - so a *group* is only a packing unit and may mix
    families with no downstream effect.

    Per epoch (built identically on every rank from seed + epoch, then sliced by rank):
      1. Each family's images are shuffled and cut into full k-sized groups (large families stay
         concentrated, giving rich episodes). The sub-k remainder becomes a "leftover" chunk.
      2. Leftover chunks (including whole small families) are bin-packed into k-slot groups by
         first-fit-decreasing, packing several small families together instead of duplicating each up
         to k - so a 1-tile family appears once per epoch, not k times. Chunks are never split.
      3. Only under-full bins (mostly just the last) are padded to k by duplicating within the bin.
      4. Groups are shuffled; rank takes groups[rank::num_replicas] (all ranks get an equal
         group count), and its groups are flattened into the index stream.
    """

    def __init__(
        self,
        im_files: list[str],
        batch_size: int,
        k: int = 4,
        seed: int = 0,
        rank: int = -1,
        num_replicas: int = 1,
    ):
        """Initialize the sampler.

        Args:
            im_files (list[str]): Image file paths; family = Path(p).parent.name.
            batch_size (int): Per-rank images per batch. Must be a multiple of k.
            k (int): Images per group.
            seed (int): Base RNG seed; combined with the epoch for per-epoch shuffling.
            rank (int): Process rank for DDP (-1 or 0 for single process).
            num_replicas (int): Number of DDP replicas (world size; 1 for single process).
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of family group size k ({k}).")
        self.k = k
        self.groups_per_batch = batch_size // k  # per rank
        self.seed = seed
        self.epoch = 0
        self.num_replicas = max(1, num_replicas)
        self.rank = max(0, rank)  # -1 (single process) -> 0

        family_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, path in enumerate(im_files):
            family_to_indices[Path(path).parent.name].append(i)
        self.family_indices = dict(family_to_indices)

        # Deterministic (shuffle-independent) group count -> equal groups per rank -> per-rank length.
        pure_groups = sum(len(v) // k for v in self.family_indices.values())
        leftover_sizes = [len(v) % k for v in self.family_indices.values() if len(v) % k]
        self.total_groups = pure_groups + _ffd_bin_count(leftover_sizes, k)
        self.groups_per_rank = self.total_groups // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so each epoch gets a different shuffle."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of indices this rank yields per epoch (groups_per_rank * k)."""
        return self.groups_per_rank * self.k

    def _build_groups(self, rng: random.Random) -> list[list[int]]:
        """Build all k-sized family groups for this epoch (identical across ranks for a given rng)."""
        # 1) Per family: shuffle, emit full k-groups (pure family), collect the sub-k remainder.
        groups: list[list[int]] = []
        leftovers: list[list[int]] = []
        for idxs in self.family_indices.values():
            shuffled = idxs[:]
            rng.shuffle(shuffled)
            full = len(shuffled) - len(shuffled) % self.k
            groups.extend(shuffled[i : i + self.k] for i in range(0, full, self.k))
            if full < len(shuffled):
                leftovers.append(shuffled[full:])  # remainder, size 1..k-1

        # 2) Bin-pack leftover chunks (whole, never split) into k-slot groups, first-fit-decreasing.
        #    Shuffle first so ties in size (e.g. many 1-tile families) group differently each epoch.
        rng.shuffle(leftovers)
        leftovers.sort(key=len, reverse=True)
        bins: list[list[int]] = []
        caps: list[int] = []  # remaining capacity per bin
        for chunk in leftovers:
            for j in range(len(bins)):
                if caps[j] >= len(chunk):
                    bins[j] += chunk
                    caps[j] -= len(chunk)
                    break
            else:
                bins.append(list(chunk))
                caps.append(self.k - len(chunk))

        # 3) Pad only under-full bins to k by duplicating within the bin, then add as groups.
        for b in bins:
            if len(b) < self.k:
                b += rng.choices(b, k=self.k - len(b))
            groups.append(b)
        return groups

    def __iter__(self):
        """Yield this rank's dataset indices for the epoch (flat, group-ordered)."""
        rng = random.Random(self.seed + self.epoch)
        groups = self._build_groups(rng)
        rng.shuffle(groups)
        # Keep an equal number of whole groups per rank; slice this rank's share, then flatten.
        usable = self.groups_per_rank * self.num_replicas
        for g in groups[: usable][self.rank :: self.num_replicas]:
            yield from g
