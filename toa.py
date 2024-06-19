import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linprog, curve_fit
from scipy.spatial import Delaunay
from spaudiopy.utils import sph2cart
from typing import Optional, Iterable, List
from soxr import resample
from functools import partial


def dot2angle(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(vec1 * vec2, axis=-1), -1, 1)
    return np.arccos(dot) / np.pi * 180


def angle2weight(angle: np.ndarray, std: float = 8) -> np.ndarray:
    return np.exp(-angle / std)


def dot2weight(vec1: np.ndarray, vec2: np.ndarray, std: float = 8) -> np.ndarray:
    return angle2weight(dot2angle(vec1, vec2), std)


def stereographic_projection(points: np.ndarray):
    """Projects points on a unit sphere to a flat plane using stereographic projection.

    Parameters
    ----------
    points : ndarray
        N x 3 array of points.

    Returns
    -------
    points_proj : ndarray
        N x 2 array of projected points.
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    N, _ = points.shape

    # project points to a plane
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    points_proj = np.zeros((N, 2))
    points_proj[:, 0] = x / (1 - z)
    points_proj[:, 1] = y / (1 - z)

    return points_proj


def edgy(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert differences.dtype == np.int64, "differences must be int"
    M = edges.shape[0]
    N = np.max(edges) + 1

    vals = np.concatenate(
        (np.ones((M,), dtype=np.int64), -np.ones((M,), dtype=np.int64))
    )
    rows = np.tile(np.arange(M), 2)
    cols = np.concatenate((edges[:, 1], edges[:, 0]))

    if weights is None:
        weights = np.ones((M,), dtype=np.int64)

    targets = differences

    num_k = M

    A_eq = csr_matrix(
        (
            np.concatenate((vals, np.ones(num_k), -np.ones(num_k))).astype(np.int64),
            (
                np.concatenate((rows, np.tile(np.arange(num_k), 2))),
                np.concatenate((cols, np.arange(2 * num_k) + N)),
            ),
        ),
        shape=(num_k, N + 2 * num_k),
    )

    c = np.concatenate((np.zeros((N,), dtype=np.int64), weights, weights))

    b_eq = targets

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, integrality=1)
    if res.x is None:
        raise ValueError("No solution found")
    m = res.x[:N]
    return np.round(m).astype(np.int64)


def toa(
    hrir: np.ndarray,
    xyz: np.ndarray,
    sr: int,
    oversampling: int = 10,
    theta: float = 8.0,
    verbose: bool = True,
) -> np.ndarray:
    if oversampling > 1:
        hrir = resample(
            hrir.reshape(-1, hrir.shape[-1]).T, sr, sr * oversampling
        ).T.reshape(hrir.shape[0], hrir.shape[1], -1)
        sr = sr * oversampling

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    hrir = hrir / np.linalg.norm(hrir, axis=2, keepdims=True)

    N, _, T = hrir.shape

    if N > 3:
        hull = Delaunay(stereographic_projection(-xyz), qhull_options="QJ")
        hull_simplices = hull.simplices

        tmp = np.stack(
            [hull_simplices, np.roll(hull_simplices, 1, axis=1)], axis=2
        ).reshape(-1, 2)
        sphere_edges = np.unique(
            np.where(tmp[:, None, 0] < tmp[:, None, 1], tmp, tmp[:, ::-1]),
            axis=0,
        )
    else:
        sphere_edges = np.array([[0, 1], [1, 2], [2, 0]])
        hull_simplices = np.array([[0, 1, 2]])

    num_unique_nodes = len(np.unique(sphere_edges.ravel()))
    assert num_unique_nodes == N, f"expected {N} nodes, but got {num_unique_nodes}"

    if verbose:
        print(f"Number of sphere simplices: {len(hull_simplices)}")
        print(f"Number of sphere edges: {len(sphere_edges)}")

    hrtf = np.fft.rfft(hrir, n=T * 2, axis=-1)
    cross_corr = np.fft.irfft(hrtf[:, 1] * hrtf[:, 0].conj(), axis=-1)
    cross_diff = np.argmax(cross_corr, axis=-1)
    cross_diff_max_corr = np.take_along_axis(
        cross_corr, cross_diff[:, None], axis=-1
    ).squeeze()
    cross_diff = (cross_diff + T) % (2 * T) - T

    if verbose:
        print(
            f"Cross correlation: max={cross_diff_max_corr.max()}, min={cross_diff_max_corr.min()}"
        )
        print(
            f"Cross delay: max={cross_diff.max() / sr * 1000} ms, min={cross_diff.min() / sr * 1000} ms"
        )

    # compute grid correlation
    left_corr = np.fft.irfft(
        hrtf[sphere_edges[:, 1], 0] * hrtf.conj()[sphere_edges[:, 0], 0], axis=-1
    )
    left_grid_diff = np.argmax(left_corr, axis=-1)
    left_grid_diff_max_corr = np.take_along_axis(
        left_corr, left_grid_diff[:, None], axis=-1
    ).squeeze()
    left_grid_diff = (left_grid_diff + T) % (2 * T) - T

    if verbose:
        print(
            f"Left grid correlation: max={left_grid_diff_max_corr.max()}, min={left_grid_diff_max_corr.min()}"
        )
        print(
            f"Left grid delay: max={left_grid_diff.max() / sr * 1000} ms, min={left_grid_diff.min() / sr * 1000} ms"
        )

    right_corr = np.fft.irfft(
        hrtf[sphere_edges[:, 1], 1] * hrtf.conj()[sphere_edges[:, 0], 1], axis=-1
    )
    right_grid_diff = np.argmax(right_corr, axis=-1)
    right_grid_diff_max_corr = np.take_along_axis(
        right_corr, right_grid_diff[:, None], axis=-1
    ).squeeze()
    right_grid_diff = (right_grid_diff + T) % (2 * T) - T

    if verbose:
        print(
            f"Right grid correlation: max={right_grid_diff_max_corr.max()}, min={right_grid_diff_max_corr.min()}"
        )
        print(
            f"Right grid delay: max={right_grid_diff.max() / sr * 1000} ms, min={right_grid_diff.min() / sr * 1000} ms"
        )

    cross_weights = dot2weight(xyz, xyz * np.array([1, -1, 1]), std=theta)
    left_grid_weights = dot2weight(
        xyz[sphere_edges[:, 0]], xyz[sphere_edges[:, 1]], std=theta
    )
    right_grid_weights = left_grid_weights

    edges = np.concatenate(
        (
            np.array([[i, i + N] for i in range(N)]),
            sphere_edges,
            sphere_edges + N,
        ),
        axis=0,
    )
    differences = np.concatenate((cross_diff, left_grid_diff, right_grid_diff))
    weights = np.concatenate((cross_weights, left_grid_weights, right_grid_weights))

    m = edgy(edges, differences, weights=weights)

    toa = m.reshape((2, N)).T

    toa = toa - toa.mean()

    if oversampling > 1:
        toa = toa / oversampling

    return toa


def toa_model(P, r, az0, az1, incli0, incli1, delta, sr):
    E = np.stack(sph2cart([az0, az1], [incli0, incli1], [r, r]), 1)
    R = np.sqrt(np.sum(P**2, 1))

    PE_cos = P @ E.T / R[:, None] / np.sqrt(np.sum(E**2, 1))
    PE_cos = np.clip(PE_cos, -1, 1)
    path_length = np.where(
        PE_cos > 0, R[:, None] - r * PE_cos, R[:, None] + r * np.arcsin(-PE_cos)
    )
    y = path_length / 343 * sr + delta
    return y


def get_rigid_params(toa, xyz, sr, verbose=True):
    assert toa.shape[0] == xyz.shape[0]

    # def toa_func(P, r, az0, az1, incli0, incli1, delta):
    #     return toa_model(P, r, az0, az1, incli0, incli1, delta, sr).ravel()

    # popt, pcov = curve_fit(
    #     toa_func,
    #     xyz,
    #     toa.ravel(),
    #     p0=(0.09, 0.5 * np.pi, 1.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0),
    #     bounds=(
    #         [0, 0, np.pi, 0, 0, -np.inf],
    #         [0.2, np.pi, 2 * np.pi, np.pi, np.pi, np.inf],
    #     ),
    # )

    def toa_func(P, r, delta):
        return toa_model(
            P, r, 0.5 * np.pi, 1.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, delta, sr
        ).ravel()

    popt, _ = curve_fit(
        toa_func,
        xyz,
        toa.ravel(),
        p0=(0.09, 0),
        bounds=(
            [0, -np.inf],
            [0.2, np.inf],
        ),
    )
    rigid_toa = toa_func(xyz, *popt).reshape(toa.shape)
    popt = (popt[0], 0.5 * np.pi, 1.5 * np.pi, 0.52 * np.pi, 0.52 * np.pi, *popt[1:])
    delay = rigid_toa.mean() / sr - popt[-1] / sr

    if verbose:
        print(f"Rigid sphere radius: {popt[0] * 100} cm")
        print(f"IR offset: {popt[-2] / sr * 1000} ms")
        print(
            f"Left ear position: {popt[1] / np.pi * 180}, {popt[3] / np.pi * 180} (az/co)"
        )
        print(
            f"Right ear position: {popt[2] / np.pi * 180 - 360}, {popt[4] / np.pi * 180} (az/co) ",
        )
        print(f"Average delay: {delay * 1000} ms")

    return {
        k: v for k, v in zip(["r", "az0", "az1", "incli0", "incli1", "delta"], popt)
    }
