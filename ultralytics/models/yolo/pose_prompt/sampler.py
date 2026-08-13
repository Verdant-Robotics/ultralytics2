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

    Cluster-aware grouping (why it is not a plain shuffle):
      The ABC loss can only learn to *aggregate* a cluster when a batch holds >=2 members of it
      (a lone member makes the query equal its own prototype - trivial - and teaches only
      separation, never within-cluster similarity). Clusters are spread thinly across a family's
      tiles, so a random k-tiles-per-group grouping rarely co-locates two members of the same
      (rare) cluster. When per-tile clusters are supplied, each family is grouped to maximise
      within-cluster co-occurrence instead:
        1. Order the family's tiles rarest-cluster-first: for each cluster present in >=2 tiles
           (in increasing tile-frequency), emit its not-yet-placed tiles consecutively; a cluster
           in a single tile is skipped (nothing to co-locate). This packs members of the same rare
           cluster next to each other.
        2. Cut the ordered tiles into HALF-groups of size k//2, then randomly pair half-groups of
           the SAME family into full k-groups. The half-group granularity + per-epoch pairing shuffle
           keep the grouping stochastic (a cluster co-occurs with different tiles each epoch) while
           still concentrating each family into k-tile groups (so its episode stays rich).
      One pass over the data is still exactly one epoch - tiles are grouped judiciously, never
      oversampled.

    Per epoch (built identically on every rank from seed + epoch, then sliced by rank):
      1. Each family is ordered (cluster-aware) and cut into k//2 half-groups; full (size-k//2)
         half-groups are shuffled and paired within the family into k-groups. A family's leftover
         half-group (odd count) and its short tail half-group become "residual" chunks.
      2. Residual chunks (including whole small families) are bin-packed into k-slot groups by
         first-fit-decreasing, packing several together instead of duplicating each up to k - so a
         1-tile family appears once per epoch, not k times. Chunks are never split.
      3. Only under-full bins (mostly just the last) are padded to k by duplicating within the bin.
      4. Groups are shuffled; rank takes groups[rank::num_replicas] (all ranks get an equal
         group count), and its groups are flattened into the index stream.
    """

    def __init__(
        self,
        im_files: list[str],
        batch_size: int,
        clusters: list[set[int]] | None = None,
        k: int = 4,
        seed: int = 0,
        rank: int = -1,
        num_replicas: int = 1,
    ):
        """Initialize the sampler.

        Args:
            im_files (list[str]): Image file paths; family = Path(p).parent.name.
            batch_size (int): Per-rank images per batch. Must be a multiple of k.
            clusters (list[set[int]] | None): Per-tile set of nonzero cluster ids, aligned
                with im_files by index. Enables cluster-aware grouping; None falls back to a plain
                per-family shuffle.
            k (int): Images per group. Must be even (grouping works in k//2 half-groups).
            seed (int): Base RNG seed; combined with the epoch for per-epoch shuffling.
            rank (int): Process rank for DDP (-1 or 0 for single process).
            num_replicas (int): Number of DDP replicas (world size; 1 for single process).
        """
        if batch_size % k != 0:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of family group size k ({k}).")
        if k % 2 != 0:
            raise ValueError(f"family group size k ({k}) must be even (grouping uses k//2 half-groups).")
        self.k = k
        self.half = k // 2
        self.clusters = clusters
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
        # Each family of n tiles yields F = n // half full half-groups (+ a size-(n % half) tail).
        # Full half-groups pair within the family into F // 2 k-groups; an odd full half-group (size
        # half) and the tail (size n % half) are residual chunks that get cross-family bin-packed.
        full_pairs = 0
        residual_sizes: list[int] = []
        for v in self.family_indices.values():
            n = len(v)
            full_halves = n // self.half
            tail = n % self.half
            full_pairs += full_halves // 2
            if full_halves % 2:
                residual_sizes.append(self.half)
            if tail:
                residual_sizes.append(tail)
        self.total_groups = full_pairs + _ffd_bin_count(residual_sizes, k)
        self.groups_per_rank = self.total_groups // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so each epoch gets a different shuffle."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of indices this rank yields per epoch (groups_per_rank * k)."""
        return self.groups_per_rank * self.k

    def _family_halves(self, idxs: list[int], rng: random.Random) -> list[list[int]]:
        """Order a family's tiles (cluster-aware) and cut them into k//2-sized half-groups.

        Every tile lands in exactly one half-group, so the count is always ceil(len(idxs) / half)
        regardless of the ordering (cluster logic only changes *which* tiles share a half-group).
        """
        if self.clusters is not None:
            # cluster -> tiles containing it; process rarest first, skip single-tile clusters.
            cluster_tiles: dict[int, list[int]] = defaultdict(list)
            for t in idxs:
                for c in self.clusters[t]:
                    cluster_tiles[c].append(t)
            ordered: list[int] = []
            seen: set[int] = set()
            for c in sorted(cluster_tiles, key=lambda c: len(cluster_tiles[c])):
                if len(cluster_tiles[c]) <= 1:
                    continue  # a lone member cannot co-occur with another - placement is irrelevant
                pool = [t for t in cluster_tiles[c] if t not in seen]
                rng.shuffle(pool)
                for t in pool:
                    ordered.append(t)
                    seen.add(t)
            rest = [t for t in idxs if t not in seen]
            rng.shuffle(rest)
            ordered.extend(rest)
        else:
            ordered = idxs[:]
            rng.shuffle(ordered)
        return [ordered[i : i + self.half] for i in range(0, len(ordered), self.half)]

    def _build_groups(self, rng: random.Random) -> list[list[int]]:
        """Build all k-sized family groups for this epoch (identical across ranks for a given rng)."""
        # 1) Per family: cluster-aware half-groups; pair full half-groups within the family into
        #    k-groups; leftover full half-group (odd count) and the short tail become residual chunks.
        groups: list[list[int]] = []
        residual: list[list[int]] = []
        for idxs in self.family_indices.values():
            halves = self._family_halves(idxs, rng)
            full = [hg for hg in halves if len(hg) == self.half]
            tail = [hg for hg in halves if len(hg) != self.half]  # 0 or 1 short half-group
            rng.shuffle(full)
            paired = len(full) - len(full) % 2
            for i in range(0, paired, 2):
                groups.append(full[i] + full[i + 1])
            if len(full) % 2:
                residual.append(full[-1])
            residual.extend(tail)

        # 2) Bin-pack residual half-group chunks (whole, never split) into k-slot groups, FFD.
        #    Shuffle first so ties in size group differently each epoch.
        rng.shuffle(residual)
        residual.sort(key=len, reverse=True)
        bins: list[list[int]] = []
        caps: list[int] = []  # remaining capacity per bin
        for chunk in residual:
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
        for g in groups[:usable][self.rank :: self.num_replicas]:
            yield from g
