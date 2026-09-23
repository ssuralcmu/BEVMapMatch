from __future__ import annotations

import numpy as np


def project_point(homography: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = homography @ np.asarray([x, y, 1.0], dtype=np.float64)
    if abs(point[2]) < 1e-12:
        raise ValueError("Degenerate homography projection")
    return float(point[0] / point[2]), float(point[1] / point[2])


def map_query_center(
    homography: np.ndarray, query_shape: tuple[int, int], reference_shape: tuple[int, int]
) -> tuple[float, float]:
    """Map the query-image center into the 3x3 reference crop.

    MatchAnything may return either transform direction. The direction whose
    projected corners are plausible in the reference frame is selected.
    """
    height, width = query_shape
    ref_height, ref_width = reference_shape
    center = ((width - 1) / 2, (height - 1) / 2)

    def outside_score(matrix: np.ndarray) -> int:
        corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        projected = [project_point(matrix, x, y) for x, y in corners]
        return sum(x < -0.5 * ref_width or x > 1.5 * ref_width or y < -0.5 * ref_height or y > 1.5 * ref_height for x, y in projected)

    direct = np.asarray(homography, dtype=np.float64)
    inverse = np.linalg.inv(direct)
    chosen = inverse if outside_score(inverse) < outside_score(direct) else direct
    return project_point(chosen, *center)

