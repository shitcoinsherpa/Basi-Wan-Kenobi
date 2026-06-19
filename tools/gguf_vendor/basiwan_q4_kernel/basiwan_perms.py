"""Generate Marlin's _perm, _scale_perm, _scale_perm_single (reproduced from
IST-DASLab/marlin/marlin/__init__.py:_get_perms, Apache-2.0).

Run as a script to dump the numerical tensors for verification:
    python basiwan_perms.py
"""
from __future__ import annotations

import numpy as np
import torch


def get_perms() -> tuple[torch.Tensor, list[int], list[int]]:
    """Return (perm: 1024-element int64 tensor, scale_perm: 64 ints, scale_perm_single: 32 ints).

    Reproduced verbatim from Marlin upstream.
    """
    perm: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_np = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_np = perm_np.reshape((-1, 8))[:, interleave].ravel()
    perm_t = torch.from_numpy(perm_np).long()

    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
        )

    return perm_t, scale_perm, scale_perm_single


if __name__ == "__main__":
    perm, scale_perm, scale_perm_single = get_perms()
    print(f"_perm.shape = {tuple(perm.shape)}  (expected: (1024,))")
    print(f"_perm[:32]  = {perm[:32].tolist()}")
    print(f"_perm[-16:] = {perm[-16:].tolist()}")
    print(f"_perm.min() = {perm.min().item()}  _perm.max() = {perm.max().item()}")
    print(f"_perm is permutation of range(0, {perm.numel()}): {sorted(perm.tolist()) == list(range(perm.numel()))}")
    print()
    print(f"_scale_perm len = {len(scale_perm)}  (expected: 64)")
    print(f"_scale_perm = {scale_perm}")
    print()
    print(f"_scale_perm_single len = {len(scale_perm_single)}  (expected: 32)")
    print(f"_scale_perm_single = {scale_perm_single}")
