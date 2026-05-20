"""Train/val split helpers for annotated frames."""

from __future__ import annotations

import random

from src.lab.annotation_store import AnnotationStore, BBox


def assign_splits(store: AnnotationStore, val_ratio: float, seed: int = 42) -> None:
    indices = store.annotated_frame_indices()
    if not indices:
        return
    rng = random.Random(seed)
    shuffled = indices[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])
    for fi in indices:
        fa = store.get_frame(fi)
        split = "val" if fi in val_set else "train"
        for b in fa.boxes:
            b.split = split
    store.save()


def split_counts(store: AnnotationStore) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        "train": {"person": 0, "car": 0, "frames": 0},
        "val": {"person": 0, "car": 0, "frames": 0},
    }
    for fi in store.annotated_frame_indices():
        fa = store.get_frame(fi)
        splits_seen: set[str] = set()
        for b in fa.boxes:
            sp = b.split if b.split in counts else "train"
            if b.class_name in counts[sp]:
                counts[sp][b.class_name] += 1
            splits_seen.add(sp)
        for sp in splits_seen:
            counts[sp]["frames"] += 1
    return counts
