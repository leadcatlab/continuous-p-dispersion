"""p-grids for the experiment sweeps."""

from typing import Dict, List

# Dense at small p, then decade-ish steps where runs get expensive.
STANDARD: List[int] = list(range(1, 11)) + [20, 30, 40, 50, 60, 70, 80, 90, 100]


def tetrahedral_numbers(limit: int) -> List[int]:
    """T(n) = n(n+1)(n+2)/6 -- 1, 4, 10, 20, 35, 56, 84, ... up to `limit`.

    A regular tetrahedron packs these counts symmetrically (n layers of a
    triangular lattice), so the optimum sits at a visibly regular configuration
    and the D(p) curve should show a distinct step pattern across them.
    """
    out, n = [], 1
    while True:
        t = n * (n + 1) * (n + 2) // 6
        if t > limit:
            return out
        out.append(t)
        n += 1


# Tetrahedron gets its own grid: small p plus the tetrahedral numbers.
TETRAHEDRON: List[int] = sorted(set(range(1, 11)) | set(tetrahedral_numbers(100)))

GRIDS: Dict[str, List[int]] = {
    "cube": STANDARD,
    "h-box": STANDARD,
    "pointy-star": STANDARD,
    "cage": STANDARD,
    "tetrahedron": TETRAHEDRON,
}


def grid_for(container: str) -> List[int]:
    return GRIDS.get(container, STANDARD)
