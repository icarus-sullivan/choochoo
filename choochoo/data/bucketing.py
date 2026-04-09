"""Resolution and aspect-ratio bucketing to minimize padding waste."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Sampler


class ResolutionBucketer:
    """Groups samples into resolution buckets to minimize padding.

    For each sample, finds the closest supported resolution bucket,
    then groups samples by bucket for efficient batching.
    """

    def __init__(
        self,
        max_resolution: int = 1024,
        min_resolution: int = 256,
        step: int = 64,
        no_upscale: bool = True,
    ):
        self.max_res = max_resolution
        self.min_res = min_resolution
        self.step = step
        self.no_upscale = no_upscale
        self._buckets = self._generate_buckets()

    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """Generate (H, W) bucket pairs that maintain roughly equal pixel count."""
        target_pixels = self.max_res * self.max_res
        buckets = set()

        sizes = range(self.min_res, self.max_res + self.step, self.step)
        for h in sizes:
            for w in sizes:
                if h * w <= target_pixels * 1.1:  # Allow slight overage
                    buckets.add((h, w))

        return sorted(buckets)

    def find_bucket(self, h: int, w: int) -> Tuple[int, int]:
        """Find the best bucket for an image of size (h, w)."""
        if self.no_upscale:
            valid = [(bh, bw) for bh, bw in self._buckets if bh <= h and bw <= w]
        else:
            valid = self._buckets

        if not valid:
            return (self.min_res, self.min_res)

        # Minimize resize distortion: find bucket with closest aspect ratio
        # and largest area that fits
        orig_ar = w / h
        best = min(
            valid,
            key=lambda bk: abs(bk[1] / bk[0] - orig_ar) + abs(bk[0] * bk[1] - h * w) / (h * w + 1)
        )
        return best

    def assign_buckets(
        self, sizes: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], List[int]]:
        """Assign indices to buckets given a list of (h, w) per sample."""
        bucket_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, (h, w) in enumerate(sizes):
            bucket = self.find_bucket(h, w)
            bucket_map[bucket].append(idx)
        return dict(bucket_map)


class BucketedBatchSampler(Sampler):
    """Sampler that yields batches from the same resolution bucket.

    Minimizes wasted compute from padding by ensuring all samples
    in a batch share the same spatial dimensions.

    Smart sampling: uses bucket-proportional weighting so that small buckets
    are sampled at their natural rate rather than being over-represented per
    epoch. Each epoch visits all buckets proportionally to size, preventing
    a tiny high-res bucket from being repeated as often as a large standard
    bucket and causing overfitting on that slice.
    """

    def __init__(
        self,
        bucket_map: Dict[Tuple[int, int], List[int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        balanced: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.bucket_map = bucket_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.balanced = balanced
        self.generator = generator

        self._batches = self._build_batches()

    def _build_batches(self) -> List[List[int]]:
        all_batches: List[List[int]] = []

        if self.balanced:
            # Proportional interleaving: sample buckets proportionally to their
            # size so large buckets get visited more often than small ones.
            all_batches = self._build_proportional_batches()
        else:
            for bucket, indices in self.bucket_map.items():
                all_batches.extend(self._slice_bucket(indices))

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=self.generator).tolist()
            all_batches = [all_batches[i] for i in perm]

        return all_batches

    def _build_proportional_batches(self) -> List[List[int]]:
        """Build batches weighted by bucket size to avoid overfitting small buckets."""
        total = sum(len(v) for v in self.bucket_map.values())
        if total == 0:
            return []

        # Each bucket contributes ceil(size / batch_size) batches; the interleave
        # order is determined by bucket weight so that, per epoch, a bucket with
        # 10% of the data produces ~10% of the batches.
        bucket_batches: Dict[Tuple[int, int], List[List[int]]] = {}
        for bucket, indices in self.bucket_map.items():
            bucket_batches[bucket] = self._slice_bucket(indices)

        # Flatten with bucket-proportional probability — achieve this by
        # concatenating sliced-bucket lists sorted by their natural size.
        combined: List[List[int]] = []
        for bucket in sorted(bucket_batches, key=lambda b: -len(bucket_batches[b])):
            combined.extend(bucket_batches[bucket])
        return combined

    def _slice_bucket(self, indices: List[int]) -> List[List[int]]:
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            indices = [indices[i] for i in perm]
        batches = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
